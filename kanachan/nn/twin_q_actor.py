import math
import torch
from torch import Tensor
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tensordict import TensorDict
from kanachan.constants import NUM_TYPES_OF_ACTIONS


class TwinQActor(nn.Module):
    def __init__(self, model0: nn.Module, model1: nn.Module) -> None:
        super().__init__()

        if isinstance(model0, DistributedDataParallel):
            self.model0 = model0.module
        else:
            self.model0 = model0
        if isinstance(model1, DistributedDataParallel):
            self.model1 = model1.module
        else:
            self.model1 = model1

    def forward(
            self,
            sparse: Tensor,
            numeric: Tensor,
            progression: Tensor,
            candidates: Tensor,
    ) -> Tensor:
        batch_size = sparse.size(0)

        td = TensorDict(
            {
                "sparse": sparse,
                "numeric": numeric,
                "progression": progression,
                "candidates": candidates,
            },
            batch_size,
        )

        self.model0(td)
        q0: Tensor = td["action_value"]
        q0 = q0.detach().clone()

        self.model1(td)
        q1: Tensor = td["action_value"]
        q1 = q1.detach().clone()

        q = torch.maximum(q0, q1)

        mask = candidates >= NUM_TYPES_OF_ACTIONS
        q = q.masked_fill(mask, -math.inf)

        return q.argmax(1)
