#!/usr/bin/env python3

import re
import datetime
import math
from pathlib import Path
import os
import logging
import sys
from typing import Tuple, Optional, Callable
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from torch import backends
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer, SGD, Adam, RAdam
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group, all_reduce, barrier
from torch.utils.tensorboard.writer import SummaryWriter
from apex.optimizers import FusedAdam, FusedSGD, FusedLAMB
from mtadam import MTAdam
from kanachan.training.constants import NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTION_CANDIDATES
from kanachan.training.common import Dataset, is_gradient_nan, get_gradient
from kanachan.model_loader import load_model
import kanachan.training.ilql.transfer.config # pylint: disable=unused-import
from kanachan.training.iql.iterator_adaptor import IteratorAdaptor
from kanachan.training.bert.encoder import Encoder
from kanachan.training.ilql.qv_model import QVDecoder, QVModel
from kanachan.training.ilql.q_model import QModel


Annotation = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


BackwardResult = Tuple[torch.Tensor, torch.Tensor, float, float, float]


def _backward(
        *, is_multiprocess: bool, world_size: Optional[int], is_amp_enabled: bool,
        annotation: Annotation, device: torch.device, amp_dtype: torch.dtype,
        qv_model: QVModel, optimizer: Optimizer, reward: float, discount_factor: float,
        expectile: float, v_loss_scaling: float, gradient_accumulation_steps: int,
        grad_scaler: GradScaler, q_target: float, local_batch_size: int) -> BackwardResult:
    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=is_amp_enabled):
        q, v = qv_model(*(annotation[:4]))
    q: torch.Tensor
    v: torch.Tensor
    assert q.dim() == 2
    assert q.size(0) == local_batch_size
    assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
    assert v.dim() == 1
    assert v.size(0) == local_batch_size
    q = q[torch.arange(local_batch_size), annotation[4]]

    q_batch_mean = q.detach().clone().mean()
    if is_multiprocess:
        all_reduce(q_batch_mean)
        q_batch_mean /= world_size

    v_batch_mean = v.detach().clone().mean()
    if is_multiprocess:
        all_reduce(v_batch_mean)
        v_batch_mean /= world_size

    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=is_amp_enabled):
        _, vv = qv_model(*(annotation[5:9]))
    vv: torch.Tensor
    is_terminal_state = (annotation[5][:, 0] == NUM_TYPES_OF_SPARSE_FEATURES)
    vv = torch.where(is_terminal_state, torch.zeros_like(vv), vv)
    assert vv.dim() == 1
    assert vv.size(0) == local_batch_size

    q_loss = torch.square(reward + discount_factor * vv - q)
    q_loss = torch.mean(q_loss)
    v_loss = q_target - v
    v_loss = torch.where(
        v_loss < 0.0, (1.0 - expectile) * torch.square(v_loss),
        expectile * torch.square(v_loss))
    v_loss = torch.mean(v_loss)
    qv_loss = q_loss + v_loss_scaling * v_loss

    qv_batch_loss = qv_loss.detach().clone().mean()
    if is_multiprocess:
        all_reduce(qv_batch_loss)
        qv_batch_loss /= world_size

    if math.isnan(qv_loss.item()):
        raise RuntimeError('QV loss becomes NaN.')

    qv_loss /= gradient_accumulation_steps
    if not isinstance(optimizer, MTAdam):
        if grad_scaler is None:
            qv_loss.backward()
        else:
            grad_scaler.scale(qv_loss).backward()

    return q_loss, v_loss, q_batch_mean.item(), v_batch_mean.item(), qv_batch_loss.item()


def _step(
        *, qv_model: QVModel, q_loss: torch.Tensor, v_loss: torch.Tensor,
        grad_scaler: Optional[GradScaler], max_gradient_norm: float, optimizer: Optimizer,
        scheduler#: lr_scheduler.LRScheduler
        ) -> Optional[float]:
    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)
    if isinstance(optimizer, MTAdam):
        qv_gradient_norm = None
    else:
        qv_gradient = get_gradient(qv_model)
        qv_gradient_norm: float = torch.linalg.vector_norm(qv_gradient).item()
    nn.utils.clip_grad_norm_(qv_model.parameters(), max_gradient_norm, error_if_nonfinite=False)
    if isinstance(optimizer, MTAdam):
        optimizer.step((q_loss, v_loss), (1.0, 1.0), None)
    else:
        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
    if scheduler is not None:
        scheduler.step()

    return qv_gradient_norm


SnapshotWriter = Callable[
    [QModel, QModel, QModel, QModel, Optimizer, Optimizer, Optional[int]],
    None
]


def _training(
        *, is_multiprocess: bool, world_size: Optional[int], rank: Optional[int],
        is_main_process: bool, training_data: Path, num_workers: int, device: torch.device,
        dtype: torch.dtype, amp_dtype: torch.dtype, qv_model: QVModel, q_model: nn.Module,
        reward_plugin: Path, discount_factor: float, expectile: float, batch_size: int,
        v_loss_scaling: float, gradient_accumulation_steps: int, max_gradient_norm: float,
        optimizer: Optimizer, scheduler#: lr_scheduler.LRScheduler
        , snapshot_interval: int, num_samples: int,
        summary_writer: SummaryWriter, snapshot_writer: SnapshotWriter) -> None:
    start_time = datetime.datetime.now()

    is_amp_enabled = (device != 'cpu' and dtype != amp_dtype)

    # Load the reward plugin.
    with open(reward_plugin, encoding='UTF-8') as file_pointer:
        exec(file_pointer.read(), globals()) # pylint: disable=exec-used

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    def iterator_adaptor(path: Path) -> IteratorAdaptor:
        return IteratorAdaptor(path, get_reward) # type: ignore pylint: disable=undefined-variable
    dataset = Dataset(training_data, iterator_adaptor)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(num_workers >= 1),
        drop_last=is_multiprocess)

    last_snapshot = None
    if snapshot_interval > 0:
        last_snapshot = num_samples

    num_consumed_samples = 0
    batch_count = 0

    grad_scaler = None
    if is_amp_enabled and not isinstance(optimizer, MTAdam):
        grad_scaler = GradScaler()

    for annotation in data_loader:
        if num_consumed_samples < num_samples:
            num_consumed_samples += batch_size
            continue

        if is_multiprocess:
            barrier()

        if is_multiprocess:
            assert world_size is not None
            assert rank is not None
            assert batch_size % world_size == 0
            first = (batch_size // world_size) * rank
            last = (batch_size // world_size) * (rank + 1)
            annotation = tuple(x[first:last] for x in annotation)

        if device != 'cpu':
            annotation = tuple(x.cuda() for x in annotation)

        local_batch_size = annotation[0].size(0)
        world_batch_size = batch_size

        # Compute the Q target value.
        with torch.no_grad():
            q_target: torch.Tensor = q_model(*(annotation[:4]))
            assert q_target.dim() == 2
            assert q_target.size(0) == local_batch_size
            assert q_target.size(1) == MAX_NUM_ACTION_CANDIDATES
            q_target = q_target[torch.arange(local_batch_size), annotation[4]]
            assert q_target.dim() == 1
            assert q_target.size(0) == local_batch_size
            q_target = q_target.detach()

        reward = annotation[9]

        # Backprop for the QV1 source model.
        q_loss, v_loss, q_batch_mean, v_batch_mean, qv_batch_loss = _backward(
            is_multiprocess=is_multiprocess, world_size=world_size, is_amp_enabled=is_amp_enabled,
            annotation=annotation, device=device, amp_dtype=amp_dtype,
            qv_model=qv_model, optimizer=optimizer, reward=reward,
            discount_factor=discount_factor, expectile=expectile, v_loss_scaling=v_loss_scaling,
            gradient_accumulation_steps=gradient_accumulation_steps, grad_scaler=grad_scaler,
            q_target=q_target, local_batch_size=local_batch_size)

        num_samples += world_batch_size
        num_consumed_samples += world_batch_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            if isinstance(optimizer, MTAdam):
                is_grad_nan = torch.zeros((), device='cpu', dtype=torch.long)
            else:
                is_grad_nan = is_gradient_nan(qv_model) or is_gradient_nan(qv_model)
                is_grad_nan = torch.where(
                    is_grad_nan, torch.ones_like(is_grad_nan), torch.zeros_like(is_grad_nan))
                all_reduce(is_grad_nan)
            if is_grad_nan.item() >= 1:
                if is_main_process:
                    logging.warning('Skip an optimization step because of NaN in the gradient.')
                qv_gradient_norm = None
            else:
                qv_gradient_norm = _step(
                    qv_model=qv_model, q_loss=q_loss, v_loss=v_loss, grad_scaler=grad_scaler,
                    max_gradient_norm=max_gradient_norm, optimizer=optimizer, scheduler=scheduler)
                if grad_scaler is not None:
                    grad_scaler.update()

            optimizer.zero_grad()

            if is_main_process:
                if qv_gradient_norm is not None:
                    logging.info(
                        'sample = %d, QV loss = %E, QV gradient norm = %E, Q = %f, V = %f',
                        num_samples, qv_batch_loss, qv_gradient_norm, q_batch_mean, v_batch_mean)
                else:
                    logging.info(
                        'sample = %d, QV loss = %E, Q = %f, V = %f',
                        num_samples, qv_batch_loss, q_batch_mean, v_batch_mean)
                summary_writer.add_scalar('Q', q_batch_mean, num_samples)
                summary_writer.add_scalar('V', v_batch_mean, num_samples)
                summary_writer.add_scalar('QV Loss', qv_batch_loss, num_samples)
                if qv_gradient_norm is not None:
                    summary_writer.add_scalar('QV Gradient Norm', qv_gradient_norm, num_samples)
                if scheduler is not None:
                    summary_writer.add_scalar('LR', scheduler.get_last_lr()[0], num_samples)
        else:
            if is_main_process:
                logging.info(
                    'sample = %d, QV loss = %E, Q = %f, V = %f',
                    num_samples, qv_batch_loss, q_batch_mean, v_batch_mean)
                summary_writer.add_scalar('Q', q_batch_mean, num_samples)
                summary_writer.add_scalar('V', v_batch_mean, num_samples)
                summary_writer.add_scalar('QV Loss', qv_batch_loss, num_samples)

        if is_main_process and last_snapshot is not None and num_samples - last_snapshot >= snapshot_interval:
            snapshot_writer(num_samples)
            last_snapshot = num_samples

    if is_multiprocess:
        barrier()

    elapsed_time = datetime.datetime.now() - start_time

    if is_main_process:
        logging.info('A training has finished (elapsed time = %s).', elapsed_time)
        snapshot_writer()


@hydra.main(version_base=None, config_name='config')
def _main(config: DictConfig) -> None:
    if 'LOCAL_RANK' in os.environ:
        if os.environ['WORLD_SIZE'] != os.environ['LOCAL_WORLD_SIZE']:
            raise RuntimeError('Multi-node not supported.')
        world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        rank = int(os.environ['LOCAL_RANK'])
        is_multiprocess = True
        is_main_process = (rank == 0)
    else:
        world_size = None
        rank = None
        is_multiprocess = False
        is_main_process = True

    if not config.training_data.exists():
        raise RuntimeError(f'{config.training_data}: Does not exist.')
    if not config.training_data.is_file():
        raise RuntimeError(f'{config.training_data}: Not a file.')

    if config.device.type == 'cpu':
        if config.num_workers is None:
            config.num_workers = 0
        if config.num_workers < 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers.')
        if config.num_workers > 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers for CPU.')
    else:
        if config.num_workers is None:
            config.num_workers = 2
        if config.num_workers < 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers.')
        if config.num_workers == 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers for GPU.')

    if config.device.type is not None:
        match = re.search('^(?:cpu|cuda(?::\\d+)?)$', config.device.type)
        if match is None:
            raise RuntimeError(f'{config.device.type}: An invalid device.')
    elif backends.cuda.is_built():
        config.device.type = 'cuda'
    else:
        config.device.type = 'cpu'
    if config.device.type == 'cuda':
        torch.cuda.set_device(rank)

    if config.device.dtype not in ('float64', 'double', 'float32', 'float', 'float16', 'half'):
        raise RuntimeError(f'{config.device.dtype}: An invalid dtype.')
    dtype = {
        'float64': torch.float64, 'double': torch.float64,
        'float32': torch.float32, 'float': torch.float32,
        'float16': torch.float16, 'half': torch.float16
    }[config.device.dtype]

    if config.device.type == 'cpu':
        if config.device.amp_dtype is not None:
            raise RuntimeError('AMP is not supported on CPU.')
        config.device.amp_dtype = config.device.dtype
    if config.device.amp_dtype is None:
        config.device.amp_dtype = config.device.dtype
    if config.device.amp_dtype not in ('float64', 'double', 'float32', 'float', 'float16', 'half'):
        raise RuntimeError(f'{config.device.amp_dtype}: An invalid AMP dtype.')
    amp_dtype = {
        'float64': torch.float64, 'double': torch.float64,
        'float32': torch.float32, 'float': torch.float32,
        'float16': torch.float16, 'half': torch.float16
    }[config.device.amp_dtype]
    if dtype == torch.float32 and amp_dtype == torch.float64:
        raise RuntimeError(
            f'An invalid combination of `device.dtype` (`{config.device.dtype}`) and '
            f'`device.amp_dtype` (`{config.device.amp_dtype}`).')
    if dtype == torch.float16 and amp_dtype in (torch.float64, torch.float32):
        raise RuntimeError(
            f'An invalid combination of `device.dtype` (`{config.device.dtype}`) and '
            f'`device.amp_dtype` (`{config.device.amp_dtype}`).')

    if backends.cudnn.is_available():
        backends.cudnn.benchmark = True

    if config.encoder.position_encoder not in ('positional_encoding', 'position_embedding'):
        raise RuntimeError(f'{config.encoder.position_encoder}: An invalid position encoder.')

    if config.encoder.dimension < 1:
        raise RuntimeError(f'{config.encoder.dimension}: An invalid dimension for the encoder.')

    if config.encoder.num_heads < 1:
        raise RuntimeError(
            f'{config.encoder.num_heads}: An invalid number of heads for the encoder.')

    if config.encoder.dim_feedforward is None:
        config.encoder.dim_feedforward = 4 * config.encoder.dimension
    if config.encoder.dim_feedforward < 1:
        raise RuntimeError(
            f'{config.encoder.dim_feedforward}:'
            ' An invalid dimension of the feedfoward networks for the encoder.')

    if config.encoder.activation_function not in ('relu', 'gelu'):
        raise RuntimeError(
            f'{config.encoder.activation_function}: '
            'An invalid activation function for the encoder.')

    if config.encoder.dropout < 0.0 or 1.0 <= config.encoder.dropout:
        raise RuntimeError(f'{config.encoder.dropout}: An invalid dropout value for the encoder.')

    if config.encoder.num_layers < 1:
        raise RuntimeError(f'{config.encoder.num_layers}: An invalid number of encoder layers.')

    if config.encoder.load_from is not None:
        if not config.encoder.load_from.exists():
            raise RuntimeError(f'{config.encoder.load_from}: Does not exist.')
        if not config.encoder.load_from.is_file():
            raise RuntimeError(f'{config.encoder.load_from}: Not a file.')

    if config.decoder.dim_feedforward is None:
        config.decoder.dim_feedforward = config.encoder.dim_feedforward
    if config.decoder.dim_feedforward < 1:
        raise RuntimeError(
            f'{config.decoder.dim_feedforward}: '
            'An invalid dimension of the feedforward networks for the decoder.')

    if config.decoder.activation_function not in ('relu', 'gelu'):
        raise RuntimeError(
            f'{config.decoder.activation_function}: '
            'An invalid activation function for the decoder.')

    if config.decoder.dropout < 0.0 or 1.0 <= config.decoder.dropout:
        raise RuntimeError(f'{config.decoder.dropout}: An invalid dropout value for the decoder.')

    if config.decoder.num_layers < 1:
        raise RuntimeError(f'{config.decoder.num_layers}: An invalid number of decoder layers.')

    if config.decoder.load_from is not None:
        if not config.decoder.load_from.exists():
            raise RuntimeError(f'{config.decoder.load_from}: Does not exist.')
        if not config.decoder.load_from.is_file():
            raise RuntimeError(f'{config.decoder.load_from}: Not a file.')

    if config.initial_model is not None:
        if not config.initial_model.exists():
            raise RuntimeError(f'{config.initial_model}: Does not exist.')
        if not config.initial_model.is_file():
            raise RuntimeError(f'{config.initial_model}: Not a file.')

    if config.initial_model_prefix is not None:
        if config.encoder.load_from is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `encoder.load_from`.')
        if config.decoder.load_from is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `decoder.load_from`.')
        if config.initial_model is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `initial_model`.')
        if not config.initial_model_prefix.exists():
            raise RuntimeError(f'{config.initial_model_prefix}: Does not exist.')
        if not config.initial_model_prefix.is_dir():
            raise RuntimeError(f'{config.initial_model_prefix}: Not a directory.')

    if config.initial_model_index is not None:
        if config.initial_model_prefix is None:
            raise RuntimeError('`initial_model_index` must be combined with `initial_model_prefix`.')
        if config.initial_model_index < 0:
            raise RuntimeError(f'{config.initial_model_index}: An invalid initial model index.')

    num_samples = 0

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_model is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                match = re.search(
                    '^(?:qv-(?:encoder|decoder)|optimizer|lr_scheduler)(?:\\.(\\d+))?\\.pth$',
                    child)
                if match is None:
                    continue
                if match[1] is None:
                    config.initial_model_index = sys.maxsize
                    continue
                if config.initial_model_index is None or int(match[1]) > config.initial_model_index:
                    config.initial_model_index = int(match[1])
                    continue
        if config.initial_model_index is None:
            raise RuntimeError(f'{config.initial_model_prefix}: No model snapshot found.')

        if config.initial_model_index == sys.maxsize:
            config.initial_model_index = 0
            infix = ''
        else:
            num_samples = config.initial_model_index
            infix = f'.{num_samples}'

        qv_encoder_snapshot_path: Path = config.initial_model_prefix / f'qv-encoder{infix}.pth'
        if not qv_encoder_snapshot_path.exists():
            raise RuntimeError(f'{qv_encoder_snapshot_path}: Does not exist.')
        if not qv_encoder_snapshot_path.is_file():
            raise RuntimeError(f'{qv_encoder_snapshot_path}: Not a file.')

        qv_decoder_snapshot_path: Path = config.initial_model_prefix / f'qv-decoder{infix}.pth'
        if not qv_decoder_snapshot_path.exists():
            raise RuntimeError(f'{qv_decoder_snapshot_path}: Does not exist.')
        if not qv_decoder_snapshot_path.is_file():
            raise RuntimeError(f'{qv_decoder_snapshot_path}: Not a file.')

        optimizer_snapshot_path: Path = config.initial_model_prefix / f'optimizer{infix}.pth'
        if not optimizer_snapshot_path.is_file() or config.optimizer.initialize:
            optimizer_snapshot_path = None

        lr_scheduler_snapshot_path: Path = config.initial_model_prefix / f'lr_scheduler{infix}.pth'
        if optimizer_snapshot_path is None:
            lr_scheduler_snapshot_path = None
        else:
            if not lr_scheduler_snapshot_path.exists():
                raise RuntimeError(f'{lr_scheduler_snapshot_path}: Does not exist.')
            if not lr_scheduler_snapshot_path.is_file():
                raise RuntimeError(f'{lr_scheduler_snapshot_path}: Not a file.')

    if not config.reward_plugin.exists():
        raise RuntimeError(f'{config.reward_plugin}: Does not exist.')
    if not config.reward_plugin.is_file():
        raise RuntimeError(f'{config.reward_plugin}: Not a file.')

    if config.discount_factor <= 0.0 or 1.0 < config.discount_factor:
        raise RuntimeError(f'{config.discount_factor}: An invalid value for `discount_factor`.')

    if config.expectile <= 0.0 or 1.0 <= config.expectile:
        raise RuntimeError(f'{config.expectile}: An invalid value for `expectile`.')

    if config.v_loss_scaling < 0.0 or 1.0 < config.v_loss_scaling:
        raise RuntimeError(f'{config.v_loss_scaling}: An invalid value for `v_loss_scaling`.')

    if config.batch_size < 1:
        raise RuntimeError(f'{config.batch_size}: An invalid value for `batch_size`.')
    if config.batch_size % world_size != 0:
        raise RuntimeError(f'`batch_size` must be divisible by the world size ({world_size}).')

    if config.gradient_accumulation_steps < 1:
        raise RuntimeError(
            f'{config.gradient_accumulation_steps}: '
            'An invalid value for `gradient_accumulation_steps`.')
    if config.gradient_accumulation_steps >= 2 and config.optimizer.type == 'mtadam':
        raise RuntimeError('`mtadam` does not support for gradient accumulation.')

    if config.max_gradient_norm <= 0.0:
        raise RuntimeError(f'{config.max_gradient_norm}: An invalid value for `max_gradient_norm`.')

    if config.optimizer.type in ('sgd',):
        if config.optimizer.momentum is None:
            raise RuntimeError('`optimizer.momentum` must be specified for `sgd`.')
        if config.optimizer.momentum < 0.0 or 1.0 <= config.optimizer.momentum:
            raise RuntimeError(
                f'{config.optimizer.momentum}: An invalid value for `optimizer.momentum`.')
    else:
        if config.optimizer.momentum is not None:
            raise RuntimeError(f'`optimizer.momentum` is useless for `{config.optimizer.type}`.')

    if config.optimizer.type in ('sgd',):
        if config.optimizer.epsilon is not None:
            raise RuntimeError(f'`optimizer.epsilon` is useless for `{config.optimizer.type}`.')
    else:
        if config.optimizer.epsilon is None:
            if config.optimizer.type in ('adam', 'radam', 'mtadam'):
                config.optimizer.epsilon = 1.0e-8
            elif config.optimizer in ('lamb',):
                config.optimizer.epsilon = 1.0e-6
            else:
                raise NotImplementedError(config.optimizer.type)
    if config.optimizer.epsilon is not None and config.optimizer.epsilon <= 0.0:
        raise RuntimeError(f'{config.optimizer.epsilon}: An invalid value for `optimizer.epsilon`.')

    if config.optimizer.learning_rate <= 0.0:
        raise RuntimeError(
            f'{config.optimizer.learning_rate}: An invalid value for `optimizer.learning_rate`.')

    if config.optimizer.warmup_steps < 0:
        raise RuntimeError(
            f'{config.optimizer.warmup_steps}: '
            '`optimizer.warmup_steps` must be a non-negative integer.')

    if config.optimizer.annealing_steps < 0:
        raise RuntimeError(
            f'{config.optimizer.annealing_steps}: '
            '`optimizer.annealing_steps` must be a non-negative integer.')

    if config.optimizer.annealing_steps_factor <= 0:
        raise RuntimeError(
            f'{config.optimizer.annealing_steps_factor}: '
            '`optimizer.annealing_steps_factor` must be a positive integer.')

    if config.snapshot_interval < 0:
        raise RuntimeError(f'{config.snapshot_interval}: An invalid value for `snapshot_interval`.')

    experiment_path = Path(HydraConfig.get().runtime.output_dir)

    tensorboard_path = experiment_path / 'tensorboard'
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    snapshots_path = experiment_path / 'snapshots'
    snapshots_path.mkdir(parents=True, exist_ok=True)

    if is_main_process:
        if world_size is None:
            assert rank is None
            logging.info('World size: N/A (single process)')
            logging.info('Process rank: N/A (single process)')
        else:
            assert rank is not None
            logging.info('World size: %d', world_size)
            logging.info('Process rank: %d', rank)
        logging.info('Training data: %s', config.training_data)
        if num_samples > 0:
            logging.info('# of training samples consumed so far: %d', num_samples)
        logging.info('# of workers: %d', config.num_workers)
        logging.info('Device: %s', config.device.type)
        if backends.cudnn.is_available():
            logging.info('cuDNN: available')
        else:
            logging.info('cuDNN: N/A')
        logging.info('dtype: %s', dtype)
        if dtype == amp_dtype:
            logging.info('AMP dtype: (AMP is disabled)')
        else:
            logging.info('AMP dtype: %s', amp_dtype)
        logging.info('Position encoder: %s', config.encoder.position_encoder)
        logging.info('Encoder dimension: %d', config.encoder.dimension)
        logging.info('# of heads for encoder: %d', config.encoder.num_heads)
        logging.info(
            'Dimension of feedforward networks for encoder: %d', config.encoder.dim_feedforward)
        logging.info('Activation function for encoder: %s', config.encoder.activation_function)
        logging.info('Dropout for encoder: %f', config.encoder.dropout)
        logging.info('# of encoder layers: %d', config.encoder.num_layers)
        if config.encoder.load_from is not None:
            logging.info('Load encoder from: %s', config.encoder.load_from)
        if config.decoder.num_layers >= 2:
            logging.info(
                'Dimension of feedforward networks for decoder: %d', config.decoder.dim_feedforward)
        logging.info('Activation function for decoder: %s', config.decoder.activation_function)
        logging.info('Dropout for decoder: %f', config.decoder.dropout)
        logging.info('# of decoder layers: %d', config.decoder.num_layers)
        if config.decoder.load_from is not None:
            logging.info('Load decoder from: %s', config.decoder.load_from)
        if config.initial_model is not None:
            logging.info('Load model from: %s', config.initial_model)
        if config.initial_model_prefix is not None:
            logging.info('Initial model prefix: %s', config.initial_model_prefix)
            logging.info('Initlal model index: %d', config.initial_model_index)
            if config.optimizer.initialize:
                logging.info('(Will not load optimizer)')
        logging.info('Q model: %s', config.q_model)
        logging.info('Reward plugin: %s', config.reward_plugin)
        logging.info('Discount factor: %f', config.discount_factor)
        logging.info('Expectile: %f', config.expectile)
        logging.info('V loss scaling: %E', config.v_loss_scaling)
        logging.info('Checkpointing: %s', config.checkpointing)
        if world_size is None:
            logging.info('Batch size: %d', config.batch_size)
        else:
            logging.info('Local batch size: %d', config.batch_size // world_size)
            logging.info('World batch size: %d', config.batch_size)
        logging.info('# of steps for gradient accumulation: %d', config.gradient_accumulation_steps)
        logging.info(
            'Virtual batch size: %d', config.batch_size * config.gradient_accumulation_steps)
        logging.info('Norm threshold for gradient clipping: %E', config.max_gradient_norm)
        logging.info('Optimizer: %s', config.optimizer.type)
        if config.optimizer in ('sgd',):
            logging.info('Momentum factor: %f', config.optimizer.momentum)
        if config.optimizer in ('adam', 'radam', 'mtadam', 'lamb'):
            logging.info('Epsilon parameter: %E', config.optimizer.epsilon)
        logging.info('Learning rate: %E', config.optimizer.learning_rate)
        if config.optimizer.warmup_steps == 0:
            logging.info('LR warm-up: (disabled)')
        else:
            logging.info('# of steps for LR warm-up: %d', config.optimizer.warmup_steps)
        if config.optimizer.annealing_steps == 0:
            logging.info('Cosine annealing: (disabled)')
        else:
            logging.info('# of steps for cosine annealing: %d', config.optimizer.annealing_steps)
            logging.info(
                'Step factor for cosine annealing: %d', config.optimizer.annealing_steps_factor)
        if config.initial_model_prefix is not None:
            logging.info('Initial QV encoder snapshot: %s', qv_encoder_snapshot_path)
            logging.info('Initial QV decoder snapshot: %s', qv_decoder_snapshot_path)
            if optimizer_snapshot_path is not None:
                logging.info('Initial optimizer snapshot: %s', optimizer_snapshot_path)
                logging.info('Initial LR scheduler snapshot: %s', lr_scheduler_snapshot_path)
        logging.info('Experiment output: %s', experiment_path)
        if config.snapshot_interval == 0:
            logging.info('Snapshot interval: N/A')
        else:
            logging.info('Snapshot interval: %d', config.snapshot_interval)

    qv_encoder = Encoder(
        position_encoder=config.encoder.position_encoder, dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads, dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function, dropout=config.encoder.dropout,
        num_layers=config.encoder.num_layers, checkpointing=config.checkpointing,
        device=config.device.type, dtype=dtype)
    qv_decoder = QVDecoder(
        dimension=config.encoder.dimension, dim_feedforward=config.decoder.dim_feedforward,
        activation_function=config.decoder.activation_function, dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers, device=config.device.type, dtype=dtype)
    qv_model = QVModel(qv_encoder, qv_decoder)
    qv_model.to(device=config.device.type, dtype=dtype)

    q_model: nn.Module = load_model(config.q_model)
    q_model.to(device=config.device.type, dtype=dtype)
    q_model.eval()

    if config.optimizer.type == 'sgd':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            if config.device.type == 'cpu':
                return SGD(
                    model.parameters(), lr=config.optimizer.learning_rate,
                    momentum=config.optimizer.momentum)
            return FusedSGD(
                model.parameters(), lr=config.optimizer.learning_rate,
                momentum=config.optimizer.momentum)
    elif config.optimizer.type == 'adam':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            if config.device.type == 'cpu':
                return Adam(
                    model.parameters(), lr=config.optimizer.learning_rate,
                    eps=config.optimizer.epsilon)
            return FusedAdam(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    elif config.optimizer.type == 'radam':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            return RAdam(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    elif config.optimizer.type == 'mtadam':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            return MTAdam(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    elif config.optimizer.type == 'lamb':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            return FusedLAMB(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    else:
        raise NotImplementedError(config.optimizer.type)
    optimizer = construct_optimizer(qv_model)

    if config.optimizer.warmup_steps == 0:
        warmup_scheduler = None
    else:
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=config.optimizer.warmup_start_factor,
            total_iters=config.optimizer.warmup_steps)
    if config.optimizer.annealing_steps == 0:
        annealing_scheduler = None
    else:
        annealing_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, config.optimizer.annealing_steps, config.optimizer.annealing_steps_factor)
    if warmup_scheduler is None and annealing_scheduler is None:
        scheduler = None
    elif warmup_scheduler is not None and annealing_scheduler is not None:
        scheduler = lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, annealing_scheduler], [warmup_scheduler.total_iters])
    elif warmup_scheduler is None:
        assert annealing_scheduler is not None
        scheduler = annealing_scheduler
    else:
        assert warmup_scheduler is not None
        assert annealing_scheduler is None
        scheduler = warmup_scheduler

    if config.encoder.load_from is not None:
        assert config.initial_model is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(config.encoder.load_from, map_location='cpu')
        encoder_new_state_dict = {}
        for key, value in encoder_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            encoder_new_state_dict[new_key] = value
        qv_encoder.load_state_dict(encoder_new_state_dict)
        if config.device.type != 'cpu':
            qv_encoder.cuda()

    if config.decoder.load_from is not None:
        assert config.initial_model is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        decoder_state_dict = torch.load(config.decoder.load_from, map_location='cpu')
        decoder_new_state_dict = {}
        for key, value in decoder_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            decoder_new_state_dict[new_key] = value
        qv_decoder.load_state_dict(decoder_new_state_dict)
        if config.device.type != 'cpu':
            qv_decoder.cuda()

    if config.initial_model is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        model_state_dict = torch.load(config.initial_model, map_location='cpu')
        model_new_state_dict = {}
        for key, value in model_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            model_new_state_dict[new_key] = value
        qv_model.load_state_dict(model_new_state_dict)
        if config.device.type != 'cpu':
            qv_model.cuda()

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_model is None

        qv_encoder_state_dict = torch.load(qv_encoder_snapshot_path, map_location='cpu')
        qv_encoder.load_state_dict(qv_encoder_state_dict)
        if config.device.type != 'cpu':
            qv_encoder.cuda()

        qv_decoder_state_dict = torch.load(qv_decoder_snapshot_path, map_location='cpu')
        qv_decoder.load_state_dict(qv_decoder_state_dict)
        if config.device.type != 'cpu':
            qv_decoder.cuda()

        if optimizer_snapshot_path is not None:
            optimizer.load_state_dict(
                torch.load(optimizer_snapshot_path, map_location='cpu'))

    if is_multiprocess:
        init_process_group(backend='nccl')
        qv_model = DistributedDataParallel(qv_model)
        qv_model = nn.SyncBatchNorm.convert_sync_batchnorm(qv_model)

    def snapshot_writer(num_samples: Optional[int]=None) -> None:
        infix = '' if num_samples is None else f'.{num_samples}'

        torch.save(qv_encoder.state_dict(), snapshots_path / f'qv-encoder{infix}.pth')
        torch.save(qv_decoder.state_dict(), snapshots_path / f'qv-decoder{infix}.pth')
        torch.save(optimizer.state_dict(), snapshots_path / f'optimizer{infix}.pth')
        if scheduler is not None:
            torch.save(scheduler.state_dict(), snapshots_path / f'lr_scheduler{infix}.pth')

    with SummaryWriter(log_dir=tensorboard_path) as summary_writer:
        _training(
            is_multiprocess=is_multiprocess, world_size=world_size, rank=rank,
            is_main_process=is_main_process, training_data=config.training_data,
            num_workers=config.num_workers, device=config.device.type, dtype=dtype,
            amp_dtype=amp_dtype, qv_model=qv_model, q_model=q_model,
            reward_plugin=config.reward_plugin, discount_factor=config.discount_factor,
            expectile=config.expectile, batch_size=config.batch_size,
            v_loss_scaling=config.v_loss_scaling,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_gradient_norm=config.max_gradient_norm, optimizer=optimizer, scheduler=scheduler,
            snapshot_interval=config.snapshot_interval, num_samples=num_samples,
            summary_writer=summary_writer,
            snapshot_writer=snapshot_writer) # pylint: disable=missing-kwoa


if __name__ == '__main__':
    _main() # pylint: disable=no-value-for-parameter
    sys.exit(0)
