import re
import math
from pathlib import Path
import datetime
import os
import logging
import sys
from typing import Optional, Callable, Any
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.amp.grad_scaler import GradScaler
from torch.distributed import (
    init_process_group,
    broadcast,
    ReduceOp,
    all_reduce,
)
from torch.utils.tensorboard.writer import SummaryWriter
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from kanachan.constants import MAX_NUM_ACTIVE_SPARSE_FEATURES, NUM_RESULTS
import kanachan.training.core.config as _config

# pylint: disable=unused-import
import kanachan.training.ranking.config  # noqa: F401
from kanachan.training.core.bc import DataLoader
from kanachan.nn import Encoder, Decoder
from kanachan.model_loader import dump_object, dump_model
from kanachan.training.common import (
    get_distributed_environment,
    get_gradient,
    is_gradient_nan,
)


SnapshotWriter = Callable[[int | None], None]


def _train(
    *,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    training_data: Path,
    rewrite_rooms: int | None,
    rewrite_grades: int | None,
    num_workers: int,
    num_samples: int,
    network_tdm: nn.Module,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_gradient_norm: float,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    snapshot_interval: int,
    summary_writer: SummaryWriter,
    snapshot_writer: SnapshotWriter,
) -> None:
    start_time = datetime.datetime.now()

    world_size, _, local_rank = get_distributed_environment()

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    data_loader = DataLoader(
        path=training_data,
        num_skip_samples=num_samples,
        rewrite_rooms=rewrite_rooms,
        rewrite_grades=rewrite_grades,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(num_workers >= 1),
        drop_last=(world_size >= 2),
    )

    is_amp_enabled = dtype != amp_dtype
    autocast_kwargs: dict[str, Any] = {
        "device_type": device.type,
        "dtype": amp_dtype,
        "enabled": is_amp_enabled,
    }
    grad_scaler = GradScaler("cuda", enabled=is_amp_enabled)

    last_snapshot: int | None = None
    if snapshot_interval > 0:
        last_snapshot = 0

    batch_count = 0

    loss_function = nn.CrossEntropyLoss()

    for data in data_loader:
        sparse: Tensor = data["sparse"]
        assert isinstance(sparse, Tensor)
        assert sparse.device == torch.device("cpu")
        assert sparse.dtype == torch.int32
        assert sparse.dim() == 2
        assert sparse.size(0) == batch_size
        assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES

        seat = sparse[:, 6] - 71
        assert seat.device == torch.device("cpu")
        assert seat.dtype == torch.int32
        assert seat.dim() == 1
        assert seat.size(0) == batch_size

        data: TensorDict = data.to(device=device)
        with torch.autocast(**autocast_kwargs):
            network_tdm(data)

        decode: Tensor = data["decode"]
        assert isinstance(decode, Tensor)
        assert decode.device == device
        assert decode.dtype == dtype
        assert decode.dim() == 3
        assert decode.size(0) == batch_size
        assert decode.size(1) == 4
        assert decode.size(2) == 4

        decode = decode[torch.arange(batch_size), seat]
        assert decode.device == device
        assert decode.dtype == dtype
        assert decode.dim() == 2
        assert decode.size(0) == batch_size
        assert decode.size(1) == 4

        results: Tensor = data["results"]
        assert isinstance(results, Tensor)
        assert results.device == device
        assert results.dtype == torch.int32
        assert results.dim() == 2
        assert results.size(0) == batch_size
        assert results.size(1) == NUM_RESULTS

        eog_scores = results[:, 10:14]
        assert eog_scores.device == device
        assert eog_scores.dtype == torch.int32
        assert eog_scores.dim() == 2
        assert eog_scores.size(0) == batch_size
        assert eog_scores.size(1) == 4
        eog_scores = eog_scores.to(device="cpu")

        ranking: Tensor = torch.zeros(
            (batch_size,), device="cpu", dtype=torch.int64
        )
        for i in range(batch_size):
            _seat = int(seat[i].item())
            _eog_scores: list[int] = eog_scores[i].tolist()
            _eog_score = _eog_scores[_seat]
            _ranking = 0
            for s in range(_seat):
                if _eog_scores[s] >= _eog_score:
                    _ranking += 1
            for s in range(_seat + 1, 4):
                if _eog_scores[s] > _eog_score:
                    _ranking += 1
            ranking[i] = _ranking
        ranking = ranking.to(device=device)

        loss: Tensor = loss_function(decode, ranking)

        _loss = loss.detach().clone()
        if world_size >= 2:
            all_reduce(_loss, ReduceOp.AVG)
        loss_to_display = float(_loss.item())

        if math.isnan(loss_to_display):
            errmsg = "Training loss becomes NaN."
            raise RuntimeError(errmsg)

        loss = loss / gradient_accumulation_steps
        grad_scaler.scale(loss).backward()  # type: ignore

        num_samples += batch_size * world_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            is_grad_nan = is_gradient_nan(network_tdm)
            if world_size >= 2:
                all_reduce(is_grad_nan)
            if is_grad_nan.item() >= 1:
                if local_rank == 0:
                    logging.warning(
                        "Skip an optimization step "
                        "because of NaN in the gradient."
                    )
                optimizer.zero_grad()
                continue

            grad_scaler.unscale_(optimizer)
            gradient = get_gradient(network_tdm)
            # pylint: disable=not-callable
            gradient_norm = float(torch.linalg.vector_norm(gradient).item())
            nn.utils.clip_grad_norm_(
                network_tdm.parameters(),
                max_gradient_norm,
                error_if_nonfinite=False,
            )
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            if local_rank == 0:
                logging.info(
                    "sample = %d, loss = %E, gradient norm = %E",
                    num_samples,
                    loss_to_display,
                    gradient_norm,
                )
            summary_writer.add_scalar("Loss", loss_to_display, num_samples)
            summary_writer.add_scalar(
                "Gradient Norm", gradient_norm, num_samples
            )
            if scheduler is not None:
                summary_writer.add_scalar(
                    "LR", scheduler.get_last_lr()[0], num_samples
                )
        else:
            if local_rank == 0:
                logging.info(
                    "sample = %d, loss = %E", num_samples, loss_to_display
                )
            summary_writer.add_scalar("Loss", loss_to_display, num_samples)

        if (
            local_rank == 0
            and last_snapshot is not None
            and num_samples - last_snapshot >= snapshot_interval
        ):
            snapshot_writer(num_samples)
            last_snapshot = num_samples

    elapsed_time = datetime.datetime.now() - start_time

    if local_rank == 0:
        logging.info(
            "A training epoch has finished (elapsed time = %s).", elapsed_time
        )
        snapshot_writer(None)


@hydra.main(version_base=None, config_name="config")
def _main(config: DictConfig) -> None:
    (
        world_size,
        rank,
        local_rank,
        device,
        dtype,
        amp_dtype,
    ) = _config.device.validate(config)

    if not config.training_data.exists():
        errmsg = f"{config.training_data}: Does not exist."
        raise RuntimeError(errmsg)
    if not config.training_data.is_file():
        errmsg = f"{config.training_data}: Not a file."
        raise RuntimeError(errmsg)

    if isinstance(config.rewrite_rooms, str):
        config.rewrite_rooms = {
            "bronze": 0,
            "silver": 1,
            "gold": 2,
            "jade": 3,
            "throne": 4,
        }[config.rewrite_rooms]
    if config.rewrite_rooms is not None and (
        config.rewrite_rooms < 0 or 4 < config.rewrite_rooms
    ):
        errmsg = (
            f"{config.rewrite_rooms}: "
            "`rewrite_rooms` must be an integer within the range [0, 4]."
        )
        raise RuntimeError(errmsg)

    if isinstance(config.rewrite_grades, str):
        config.rewrite_grades = {
            "novice1": 0,
            "novice2": 1,
            "novice3": 2,
            "adept1": 3,
            "adept2": 4,
            "adept3": 5,
            "expert1": 6,
            "expert2": 7,
            "expert3": 8,
            "master1": 9,
            "master2": 10,
            "master3": 11,
            "saint1": 12,
            "saint2": 13,
            "saint3": 14,
            "celestial": 15,
        }[config.rewrite_grades]
    if config.rewrite_grades is not None and (
        config.rewrite_grades < 0 or 15 < config.rewrite_grades
    ):
        errmsg = (
            f"{config.rewrite_grades}: "
            "`rewrite_grades` must be an integer within the range [0, 15]."
        )
        raise RuntimeError(errmsg)

    if device.type == "cpu":
        if config.num_workers is None:
            config.num_workers = 0
        if config.num_workers < 0:
            errmsg = f"{config.num_workers}: An invalid number of workers."
            raise RuntimeError(errmsg)
        if config.num_workers > 0:
            errmsg = (
                f"{config.num_workers}: An invalid number of workers for CPU."
            )
            raise RuntimeError(errmsg)
    else:
        assert device.type == "cuda"
        if config.num_workers is None:
            config.num_workers = 2
        if config.num_workers < 0:
            errmsg = f"{config.num_workers}: An invalid number of workers."
            raise RuntimeError(errmsg)
        if config.num_workers == 0:
            errmsg = (
                f"{config.num_workers}: An invalid number of workers for GPU."
            )
            raise RuntimeError(errmsg)

    _config.encoder.validate(config)

    _config.decoder.validate(config)

    if config.initial_model_prefix is not None:
        if config.encoder.load_from is not None:
            errmsg = (
                "`initial_model_prefix` conflicts with `encoder.load_from`."
            )
            raise RuntimeError(errmsg)
        if not config.initial_model_prefix.exists():
            errmsg = f"{config.initial_model_prefix}: Does not exist."
            raise RuntimeError(errmsg)
        if not config.initial_model_prefix.is_dir():
            errmsg = f"{config.initial_model_prefix}: Not a directory."
            raise RuntimeError(errmsg)

    if config.initial_model_index is not None:
        if config.initial_model_prefix is None:
            errmsg = (
                "`initial_model_index` must be combined with "
                "`initial_model_prefix`."
            )
            raise RuntimeError(errmsg)
        if config.initial_model_index < 0:
            errmsg = (
                f"{config.initial_model_index}:"
                " An invalid initial model index."
            )
            raise RuntimeError(errmsg)

    num_samples = 0
    encoder_snapshot_path: Path | None = None
    decoder_snapshot_path: Path | None = None
    optimizer_snapshot_path: Path | None = None
    scheduler_snapshot_path: Path | None = None

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                match = re.search(
                    "^(?:encoder|decoder|optimizer|lr-scheduler)(?:\\.(\\d+))?\\.pth$",
                    child,
                )
                if match is None:
                    continue
                if match[1] is None:
                    config.initial_model_index = sys.maxsize
                    continue
                if (
                    config.initial_model_index is None
                    or int(match[1]) > config.initial_model_index
                ):
                    config.initial_model_index = int(match[1])
                    continue
        if config.initial_model_index is None:
            errmsg = f"{config.initial_model_prefix}: No model snapshot found."
            raise RuntimeError(errmsg)

        if config.initial_model_index == sys.maxsize:
            config.initial_model_index = 0
            infix = ""
        else:
            num_samples = config.initial_model_index
            infix = f".{num_samples}"

        encoder_snapshot_path = (
            config.initial_model_prefix / f"encoder{infix}.pth"
        )
        assert encoder_snapshot_path is not None
        if not encoder_snapshot_path.exists():
            errmsg = f"{encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not encoder_snapshot_path.is_file():
            errmsg = f"{encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        decoder_snapshot_path = (
            config.initial_model_prefix / f"decoder{infix}.pth"
        )
        assert decoder_snapshot_path
        if not decoder_snapshot_path.exists():
            errmsg = f"{decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not decoder_snapshot_path.is_file():
            errmsg = f"{decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        optimizer_snapshot_path = (
            config.initial_model_prefix / f"optimizer{infix}.pth"
        )
        if optimizer_snapshot_path is not None and (
            not optimizer_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            optimizer_snapshot_path = None

        scheduler_snapshot_path = (
            config.initial_model_prefix / f"lr-scheduler{infix}.pth"
        )
        if scheduler_snapshot_path is not None and (
            not scheduler_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            scheduler_snapshot_path = None

    if config.batch_size <= 0:
        errmsg = (
            f"{config.batch_size}: `batch_size` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if config.gradient_accumulation_steps < 1:
        errmsg = (
            f"{config.gradient_accumulation_steps}: "
            "`gradient_accumulation_steps` must be an integer greater than 0."
        )
        raise RuntimeError(errmsg)

    if config.max_gradient_norm <= 0.0:
        errmsg = (
            f"{config.max_gradient_norm}: "
            "`max_gradient_norm` must be a positive real value."
        )
        raise RuntimeError(errmsg)

    _config.optimizer.validate(config)

    if config.snapshot_interval < 0:
        errmsg = (
            f"{config.snapshot_interval}: "
            "`snapshot_interval` must be a non-negative integer."
        )
        raise RuntimeError(errmsg)

    output_prefix = Path(HydraConfig.get().runtime.output_dir)

    if local_rank == 0:
        _config.device.dump(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
        )

        logging.info("Training data: %s", config.training_data)
        if num_samples > 0:
            logging.info(
                "# of training samples consumed so far: %d", num_samples
            )
        if config.rewrite_rooms is not None:
            logging.info(
                "Rewrite the rooms in the training data to: %d",
                config.rewrite_rooms,
            )
        if config.rewrite_grades is not None:
            logging.info(
                "Rewrite the ranks in the training data to: %d",
                config.rewrite_grades,
            )
        logging.info("# of workers: %d", config.num_workers)

        _config.encoder.dump(config)

        _config.decoder.dump(config)

        if config.initial_model_prefix is not None:
            logging.info(
                "Initial model prefix: %s", config.initial_model_prefix
            )
            logging.info("Initlal model index: %d", config.initial_model_index)
            if config.optimizer.initialize:
                logging.info("(Will not load optimizer)")

        logging.info("Checkpointing: %s", config.checkpointing)
        logging.info("Batch size: %d", config.batch_size)
        logging.info(
            "# of steps for gradient accumulation: %d",
            config.gradient_accumulation_steps,
        )
        logging.info(
            "Norm threshold for gradient clipping: %E",
            config.max_gradient_norm,
        )

        _config.optimizer.dump(config)

        if config.initial_model_prefix is not None:
            logging.info("Initial encoder snapshot: %s", encoder_snapshot_path)
            logging.info("Initial decoder snapshot: %s", decoder_snapshot_path)
            if optimizer_snapshot_path is not None:
                logging.info(
                    "Initial optimizer snapshot: %s", optimizer_snapshot_path
                )
            if scheduler_snapshot_path is not None:
                logging.info(
                    "Initial LR scheduler snapshot: %s",
                    scheduler_snapshot_path,
                )

        logging.info("Output prefix: %s", output_prefix)
        if config.snapshot_interval == 0:
            logging.info("Snapshot interval: N/A")
        else:
            logging.info("Snapshot interval: %d", config.snapshot_interval)

    if world_size >= 2:
        init_process_group(backend="nccl")

    encoder = Encoder(
        position_encoder=config.encoder.position_encoder,
        dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads,
        dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function,
        dropout=config.encoder.dropout,
        layer_normalization=config.encoder.layer_normalization,
        num_layers=config.encoder.num_layers,
        checkpointing=config.checkpointing,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    encoder_tdm = TensorDictModule(
        encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],
        out_keys=["encode"],
    )
    decoder = Decoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        layer_normalization=config.decoder.layer_normalization,
        num_layers=config.decoder.num_layers,
        output_mode="ranking",
        noise_init_std=None,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    with torch.no_grad():
        for _param in decoder.parameters():
            _param.zero_()
    decoder_tdm = TensorDictModule(
        decoder, in_keys=["encode"], out_keys=["decode"]
    )
    network_tdm = TensorDictSequential(encoder_tdm, decoder_tdm)
    if world_size >= 2:
        network_tdm.to(device=device)
        for _param in network_tdm.parameters():
            broadcast(_param.data, src=0)
        network_tdm.to(device="cpu")

    softmax = nn.Softmax(2)
    softmax_tdm = TensorDictModule(
        softmax, in_keys=["decode"], out_keys=["ranking_probs"]
    )
    network_tdm_to_save = TensorDictSequential(
        encoder_tdm, decoder_tdm, softmax_tdm
    )

    optimizer, scheduler = _config.optimizer.create(config, network_tdm)

    if config.encoder.load_from is not None:
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(
            config.encoder.load_from, map_location="cpu", weights_only=True
        )
        encoder.load_state_dict(encoder_state_dict)

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert encoder_snapshot_path is not None
        assert decoder_snapshot_path is not None

        encoder_state_dict = torch.load(
            encoder_snapshot_path, map_location="cpu", weights_only=True
        )
        encoder.load_state_dict(encoder_state_dict)

        decoder_state_dict = torch.load(
            decoder_snapshot_path, map_location="cpu", weights_only=True
        )
        decoder.load_state_dict(decoder_state_dict)

        if optimizer_snapshot_path is not None:
            optimizer.load_state_dict(
                torch.load(
                    optimizer_snapshot_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )

        if scheduler_snapshot_path is not None:
            assert scheduler is not None
            scheduler.load_state_dict(
                torch.load(
                    scheduler_snapshot_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )

    network_tdm.requires_grad_(True)
    network_tdm.train()
    network_tdm.to(device=device, dtype=dtype)
    if world_size >= 2:
        network_tdm = DistributedDataParallel(network_tdm)
        network_tdm = nn.SyncBatchNorm.convert_sync_batchnorm(network_tdm)
        assert isinstance(network_tdm, nn.Module)

    network_tdm_to_save.requires_grad_(True)
    network_tdm_to_save.train()
    network_tdm_to_save.to(device=device, dtype=dtype)

    snapshots_path = output_prefix / "snapshots"

    def snapshot_writer(num_samples: Optional[int] = None) -> None:
        snapshots_path.mkdir(parents=True, exist_ok=True)

        infix = "" if num_samples is None else f".{num_samples}"

        torch.save(
            encoder.state_dict(), snapshots_path / f"encoder{infix}.pth"
        )
        torch.save(
            decoder.state_dict(), snapshots_path / f"decoder{infix}.pth"
        )
        torch.save(
            optimizer.state_dict(), snapshots_path / f"optimizer{infix}.pth"
        )
        if scheduler is not None:
            torch.save(
                scheduler.state_dict(),
                snapshots_path / f"lr-scheduler{infix}.pth",
            )

        network_tdm_state = dump_object(
            network_tdm_to_save,
            [
                dump_object(
                    encoder_tdm,
                    [
                        dump_model(
                            encoder,
                            [],
                            {
                                "position_encoder": config.encoder.position_encoder,
                                "dimension": config.encoder.dimension,
                                "num_heads": config.encoder.num_heads,
                                "dim_feedforward": config.encoder.dim_feedforward,
                                "activation_function": config.encoder.activation_function,
                                "dropout": config.encoder.dropout,
                                "layer_normalization": config.encoder.layer_normalization,
                                "num_layers": config.encoder.num_layers,
                                "checkpointing": config.checkpointing,
                                "device": torch.device("cpu"),
                                "dtype": dtype,
                            },
                        ),
                    ],
                    {
                        "in_keys": [
                            "sparse",
                            "numeric",
                            "progression",
                            "candidates",
                        ],
                        "out_keys": ["encode"],
                    },
                ),
                dump_object(
                    decoder_tdm,
                    [
                        dump_model(
                            decoder,
                            [],
                            {
                                "input_dimension": config.encoder.dimension,
                                "dimension": config.decoder.dimension,
                                "activation_function": config.decoder.activation_function,
                                "dropout": config.decoder.dropout,
                                "layer_normalization": config.decoder.layer_normalization,
                                "num_layers": config.decoder.num_layers,
                                "output_mode": "ranking",
                                "noise_init_std": None,
                                "device": torch.device("cpu"),
                                "dtype": dtype,
                            },
                        ),
                    ],
                    {
                        "in_keys": ["encode"],
                        "out_keys": ["decode"],
                    },
                ),
                dump_object(
                    softmax_tdm,
                    [dump_model(softmax, [2], {})],
                    {
                        "in_keys": ["decode"],
                        "out_keys": ["ranking_probs"],
                    },
                ),
            ],
            {},
        )
        torch.save(
            network_tdm_state, snapshots_path / f"model{infix}.kanachan"
        )

    tensorboard_path = output_prefix / "tensorboard"
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=tensorboard_path) as summary_writer:
        torch.autograd.set_detect_anomaly(
            False
        )  # `True` for debbing purpose only.
        _train(
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
            training_data=config.training_data,
            rewrite_rooms=config.rewrite_rooms,
            rewrite_grades=config.rewrite_grades,
            num_workers=config.num_workers,
            num_samples=num_samples,
            network_tdm=network_tdm,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_gradient_norm=config.max_gradient_norm,
            optimizer=optimizer,
            scheduler=scheduler,
            snapshot_interval=config.snapshot_interval,
            summary_writer=summary_writer,
            snapshot_writer=snapshot_writer,
        )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    _main()
    sys.exit(0)
