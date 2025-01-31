import math
import torch
from torch import Tensor, nn
from tensordict import TensorDict
from kanachan.constants import (
    NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES,
    ENCODER_WIDTH,
)
from kanachan.nn import Decoder


class QRDecoder(nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        dimension: int | None,
        activation_function: str | None,
        dropout: float | None,
        layer_normalization: bool,
        num_layers: int,
        num_qr_intervals: int,
        dueling_network: bool,
        noise_init_std: float | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if input_dimension <= 0:
            raise ValueError(input_dimension)
        if dimension is not None and dimension <= 0:
            raise ValueError(dimension)
        if activation_function not in (None, "relu", "gelu"):
            raise ValueError(activation_function)
        if dropout is not None and (dropout < 0.0 or 1.0 <= dropout):
            raise ValueError(dropout)
        if num_layers <= 0:
            raise ValueError(num_layers)
        if num_layers == 1:
            if dimension is not None:
                raise ValueError(dimension)
            if activation_function is not None:
                raise ValueError(activation_function)
            if dropout is not None:
                raise ValueError(dropout)
        if num_qr_intervals <= 0:
            raise ValueError(num_qr_intervals)
        if noise_init_std is not None and noise_init_std < 0.0:
            raise ValueError(noise_init_std)

        super().__init__()

        self.state_value_decoder_list = nn.ModuleList()
        self.advantage_decoder_list = nn.ModuleList()
        for _ in range(num_qr_intervals):
            if dueling_network:
                state_value_decoder = Decoder(
                    input_dimension=input_dimension,
                    dimension=dimension,
                    activation_function=activation_function,
                    dropout=dropout,
                    layer_normalization=layer_normalization,
                    num_layers=num_layers,
                    output_mode="state",
                    noise_init_std=noise_init_std,
                    device=device,
                    dtype=dtype,
                )
                self.state_value_decoder_list.append(state_value_decoder)

            advantage_decoder = Decoder(
                input_dimension=input_dimension,
                dimension=dimension,
                activation_function=activation_function,
                dropout=dropout,
                layer_normalization=layer_normalization,
                num_layers=num_layers,
                output_mode="candidates",
                noise_init_std=noise_init_std,
                device=device,
                dtype=dtype,
            )
            self.advantage_decoder_list.append(advantage_decoder)

    def forward(self, candidates: Tensor, encode: Tensor) -> Tensor:
        assert candidates.dim() == 2
        batch_size = candidates.size(0)
        assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
        assert encode.dim() == 3
        assert encode.size(0) == batch_size
        assert encode.size(1) == ENCODER_WIDTH

        mask = candidates >= NUM_TYPES_OF_ACTIONS

        num_qr_intervals = len(self.advantage_decoder_list)
        dueling_network = len(self.state_value_decoder_list) != 0
        assert dueling_network == (
            len(self.state_value_decoder_list) == num_qr_intervals
        )
        qs: list[Tensor] = []
        for i in range(num_qr_intervals):
            advantage_decoder = self.advantage_decoder_list[i]
            advantage: Tensor = advantage_decoder(encode)
            assert advantage.dim() == 2
            assert advantage.size(0) == batch_size
            assert advantage.size(1) == MAX_NUM_ACTION_CANDIDATES

            if dueling_network:
                advantage = advantage.masked_fill(mask, 0.0)
                advantage = advantage - advantage.mean(1, keepdim=True)
                state_value_decoder = self.state_value_decoder_list[i]
                state_value: Tensor = state_value_decoder(encode)
                assert state_value.dim() == 1
                assert state_value.size(0) == batch_size
                advantage += state_value.unsqueeze(1).expand_as(advantage)

            qs.append(advantage)

        theta = torch.stack(qs, 2)
        assert theta.dim() == 3
        assert theta.size(0) == batch_size
        assert theta.size(1) == MAX_NUM_ACTION_CANDIDATES
        assert theta.size(2) == num_qr_intervals
        theta = theta.masked_fill(
            mask.unsqueeze(2).expand_as(theta), -math.inf
        )

        return theta


def _get_a_star(source_network: nn.Module, data: TensorDict) -> Tensor:
    batch_size = int(data.batch_size[0])

    copy: TensorDict = data["next"]
    assert isinstance(copy, TensorDict)
    copy = copy.detach()
    copy = copy.clone()
    with torch.no_grad():
        source_network(copy)

    next_theta: Tensor = copy["qr_action_value"]
    assert isinstance(next_theta, Tensor)
    assert next_theta.dim() == 3
    assert next_theta.size(0) == batch_size
    assert next_theta.size(1) == MAX_NUM_ACTION_CANDIDATES
    num_qr_intervals = int(next_theta.size(2))

    next_q = torch.sum(next_theta * (1.0 / num_qr_intervals), dim=2)
    assert next_q.dim() == 2
    assert next_q.size(0) == batch_size
    assert next_q.size(1) == MAX_NUM_ACTION_CANDIDATES

    a_star = next_q.argmax(1).to(torch.int32)
    assert a_star.dim() == 1
    assert a_star.size(0) == batch_size

    return a_star


def compute_td_error(
    *,
    source_network: nn.Module,
    target_network: nn.Module | None,
    data: TensorDict,
    discount_factor: float,
    kappa: float,
) -> None:
    batch_size = int(data.batch_size[0])

    a_star = _get_a_star(source_network, data)

    copy: TensorDict = data.detach()
    copy = copy.clone()
    source_network(copy)
    data["qr_action_value"] = copy["qr_action_value"]

    _copy = TensorDict(
        {
            "sparse": data["next", "sparse"].detach().clone(),
            "numeric": data["next", "numeric"].detach().clone(),
            "progression": data["next", "progression"].detach().clone(),
            "candidates": data["next", "candidates"].detach().clone(),
        },
        batch_size=data.batch_size,
        device=data.device,
    )
    if target_network is None:
        source_network(_copy)
    else:
        with torch.no_grad():
            target_network(_copy)
    data["next", "qr_action_value"] = _copy["qr_action_value"]

    theta: Tensor = data["qr_action_value"]
    assert isinstance(theta, Tensor)
    assert theta.dim() == 3
    assert theta.size(0) == batch_size
    assert theta.size(1) == MAX_NUM_ACTION_CANDIDATES
    num_qr_intervals = int(theta.size(2))

    action: Tensor = data["action"]
    assert isinstance(action, Tensor)
    assert action.dtype == torch.int32
    assert action.dim() == 1
    assert action.size(0) == batch_size

    theta_sa = theta[torch.arange(batch_size), action]
    assert theta_sa.dim() == 2
    assert theta_sa.size(0) == batch_size
    assert theta_sa.size(1) == num_qr_intervals

    q = torch.sum(theta * (1.0 / num_qr_intervals), dim=2)
    assert q.dim() == 2
    assert q.size(0) == batch_size
    assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
    data["action_value"] = q

    next_theta: Tensor = data["next", "qr_action_value"]
    assert isinstance(next_theta, Tensor)
    assert next_theta.dim() == 3
    assert next_theta.size(0) == batch_size
    assert next_theta.size(1) == MAX_NUM_ACTION_CANDIDATES
    assert next_theta.size(2) == num_qr_intervals

    next_q = torch.sum(next_theta * (1.0 / num_qr_intervals), dim=2)
    assert next_q.dim() == 2
    assert next_q.size(0) == batch_size
    assert next_q.size(1) == MAX_NUM_ACTION_CANDIDATES
    data["next", "action_value"] = next_q

    reward: Tensor = data["next", "reward"]
    assert isinstance(reward, Tensor)
    assert reward.dim() == 1
    assert reward.size(0) == batch_size

    done: Tensor = data["next", "done"]
    assert isinstance(done, Tensor)
    assert done.dtype == torch.bool
    assert done.dim() == 1
    assert done.size(0) == batch_size

    tau_theta = torch.where(
        done.unsqueeze(1),
        reward.unsqueeze(1),
        reward.unsqueeze(1)
        + discount_factor * next_theta[torch.arange(batch_size), a_star],
    )
    assert tau_theta.dim() == 2
    assert tau_theta.size(0) == batch_size
    assert tau_theta.size(1) == num_qr_intervals

    u = tau_theta.unsqueeze(1) - theta_sa.unsqueeze(2)
    assert u.dim() == 3
    assert u.size(0) == batch_size
    assert u.size(1) == num_qr_intervals
    assert u.size(2) == num_qr_intervals

    lu = torch.where(
        torch.abs(u) <= kappa,
        0.5 * (u**2.0),
        kappa * (torch.abs(u) - 0.5 * kappa),
    )

    factor_i = torch.arange(
        0.5 / num_qr_intervals,
        1.0,
        1.0 / num_qr_intervals,
        device=u.device,
        dtype=u.dtype,
    )
    assert factor_i.dim() == 1
    assert factor_i.size(0) == num_qr_intervals
    factor_i = factor_i.unsqueeze(0).unsqueeze(2).expand_as(u).detach().clone()
    factor_i -= torch.where(u < 0.0, torch.ones_like(u), torch.zeros_like(u))

    rho = torch.abs(factor_i) * lu

    td_error = rho.mean(2).sum(1)
    assert td_error.dim() == 1
    assert td_error.size(0) == batch_size

    data["td_error"] = td_error


class QDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, qr_decode: Tensor) -> Tensor:
        assert qr_decode.dim() == 3
        # batch_size = qr_decode.size(0)
        assert qr_decode.size(1) == MAX_NUM_ACTION_CANDIDATES
        num_qr_intervals = qr_decode.size(2)

        decode = torch.sum(qr_decode * (1.0 / num_qr_intervals), 2)
        return decode
