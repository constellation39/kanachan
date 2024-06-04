from pathlib import Path
from tqdm import tqdm
import torch
from torch import Tensor
import torch.utils.data
from tensordict import TensorDictBase, TensorDict
from torchrl.data import ListStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from kanachan.constants import (
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    MAX_NUM_ROUND_SUMMARY,
    NUM_RESULTS,
)
from kanachan.training.common import get_distributed_environment
from kanachan.training.core.rl import RewardFunction
from kanachan.training.core.offline_rl.data_iterator import DataIterator
from kanachan.training.core.dataset import Dataset


_INTERNAL_BATCH_SIZE = 512


class EpisodeReplayBuffer:
    def __init__(
        self,
        *,
        training_data: Path,
        contiguous_training_data: bool,
        num_skip_samples: int,
        rewrite_rooms: int | None,
        rewrite_grades: int | None,
        get_reward: RewardFunction,
        dtype: torch.dtype,
        max_size: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
    ) -> None:
        if not training_data.exists():
            raise RuntimeError(f"{training_data}: Does not exist.")
        if not training_data.is_file():
            raise RuntimeError(f"{training_data}: Not a file.")
        if num_skip_samples < 0:
            raise ValueError(num_skip_samples)
        if rewrite_rooms is not None:
            if rewrite_rooms < 0 or rewrite_rooms > 4:
                raise ValueError(rewrite_rooms)
        if rewrite_grades is not None:
            if rewrite_grades < 0 or rewrite_grades > 15:
                raise ValueError(rewrite_grades)
        if max_size <= 0:
            raise ValueError(max_size)
        if batch_size <= 0:
            raise ValueError(batch_size)
        if batch_size > max_size:
            raise ValueError(f"{batch_size} > {max_size}")
        if num_workers < 0:
            raise ValueError(num_workers)

        world_size, _, _ = get_distributed_environment()

        dataset = Dataset(
            path=training_data,
            iterator_class=DataIterator,
            num_skip_samples=num_skip_samples,
            rewrite_rooms=rewrite_rooms,
            rewrite_grades=rewrite_grades,
        )
        self.__data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=_INTERNAL_BATCH_SIZE,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.__contiguous = contiguous_training_data
        self.__get_reward = get_reward
        self.__dtype = dtype
        storage = ListStorage(max_size=max_size)
        sampler = SamplerWithoutReplacement(drop_last=drop_last)
        self.__replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=sampler,
            batch_size=(batch_size * world_size),
        )
        self.__batch_size = batch_size
        self.__max_size = max_size
        self.__size = 0
        self.__batch_queue: list[TensorDict] = []
        self.__first_iteration = True

    def __iter__(self) -> "EpisodeReplayBuffer":
        return self

    def __next__(self) -> TensorDict:
        if len(self.__batch_queue) >= 1:
            return self.__batch_queue.pop(0)

        world_size, rank, local_rank = get_distributed_environment()

        progress: tqdm | None = None
        if self.__first_iteration:
            progress = tqdm(
                desc="Loading data to replay buffer...",
                total=self.__max_size,
                maxinterval=0.1,
                disable=(local_rank != 0),
                unit=" samples",
                smoothing=0.0,
            )
            self.__first_iteration = False

        annotations: TensorDictBase = TensorDict(
            {}, batch_size=0, device="cpu"
        )
        annotations.set(
            "sparse", torch.empty(0, device="cpu", dtype=torch.int32)
        )
        annotations.set(
            "numeric", torch.empty(0, device="cpu", dtype=torch.int32)
        )
        annotations.set(
            "progression", torch.empty(0, device="cpu", dtype=torch.int32)
        )
        annotations.set(
            "candidates", torch.empty(0, device="cpu", dtype=torch.int32)
        )
        annotations.set(
            "action", torch.empty(0, device="cpu", dtype=torch.int32)
        )
        annotations.set(
            ("next", "sparse"), torch.empty(0, device="cpu", dtype=torch.int32)
        )
        annotations.set(
            ("next", "numeric"),
            torch.empty(0, device="cpu", dtype=torch.int32),
        )
        annotations.set(
            ("next", "progression"),
            torch.empty(0, device="cpu", dtype=torch.int32),
        )
        annotations.set(
            ("next", "candidates"),
            torch.empty(0, device="cpu", dtype=torch.int32),
        )
        annotations.set(
            ("next", "round_summary"),
            torch.empty(0, device="cpu", dtype=torch.int32),
        )
        annotations.set(
            ("next", "results"),
            torch.empty(0, device="cpu", dtype=torch.int32),
        )
        annotations.set(
            ("next", "end_of_round"),
            torch.empty(0, device="cpu", dtype=torch.bool),
        )
        annotations.set(
            ("next", "end_of_game"),
            torch.empty(0, device="cpu", dtype=torch.bool),
        )
        annotations.set(
            ("next", "done"), torch.empty(0, device="cpu", dtype=torch.bool)
        )

        for data in self.__data_loader:
            assert isinstance(data, list)
            assert len(data) == 13

            sparse: Tensor = data[0]
            assert isinstance(sparse, Tensor)
            assert sparse.device == torch.device("cpu")
            assert sparse.dtype == torch.int32
            assert sparse.dim() == 2
            _internal_batch_size = sparse.size(0)
            assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES

            numeric: Tensor = data[1]
            assert isinstance(numeric, Tensor)
            assert numeric.device == torch.device("cpu")
            assert numeric.dtype == torch.int32
            assert numeric.dim() == 2
            assert numeric.size(0) == _internal_batch_size
            assert numeric.size(1) == NUM_NUMERIC_FEATURES

            progression: Tensor = data[2]
            assert isinstance(progression, Tensor)
            assert progression.device == torch.device("cpu")
            assert progression.dtype == torch.int32
            assert progression.dim() == 2
            assert progression.size(0) == _internal_batch_size
            assert progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES

            candidates: Tensor = data[3]
            assert isinstance(candidates, Tensor)
            assert candidates.dtype == torch.int32
            assert candidates.device == torch.device("cpu")
            assert candidates.dim() == 2
            assert candidates.size(0) == _internal_batch_size
            assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES

            action: Tensor = data[4]
            assert isinstance(action, Tensor)
            assert action.dtype == torch.int32
            assert action.dim() == 1
            assert action.size(0) == _internal_batch_size

            next_sparse: Tensor = data[5]
            assert isinstance(next_sparse, Tensor)
            assert next_sparse.device == torch.device("cpu")
            assert next_sparse.dtype == torch.int32
            assert next_sparse.dim() == 2
            assert next_sparse.size(0) == _internal_batch_size
            assert next_sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES

            next_numeric: Tensor = data[6]
            assert isinstance(next_numeric, Tensor)
            assert next_numeric.device == torch.device("cpu")
            assert next_numeric.dtype == torch.int32
            assert next_numeric.dim() == 2
            assert next_numeric.size(0) == _internal_batch_size
            assert next_numeric.size(1) == NUM_NUMERIC_FEATURES

            next_progression: Tensor = data[7]
            assert isinstance(next_progression, Tensor)
            assert next_progression.device == torch.device("cpu")
            assert next_progression.dtype == torch.int32
            assert next_progression.dim() == 2
            assert next_progression.size(0) == _internal_batch_size
            assert (
                next_progression.size(1) == MAX_LENGTH_OF_PROGRESSION_FEATURES
            )

            next_candidates: Tensor = data[8]
            assert isinstance(next_candidates, Tensor)
            assert next_candidates.device == torch.device("cpu")
            assert next_candidates.dtype == torch.int32
            assert next_candidates.dim() == 2
            assert next_candidates.size(0) == _internal_batch_size
            assert next_candidates.size(1) == MAX_NUM_ACTION_CANDIDATES

            round_summary: Tensor = data[9]
            assert isinstance(round_summary, Tensor)
            assert round_summary.device == torch.device("cpu")
            assert round_summary.dtype == torch.int32
            assert round_summary.dim() == 2
            assert round_summary.size(0) == _internal_batch_size
            assert round_summary.size(1) == MAX_NUM_ROUND_SUMMARY

            results: Tensor = data[10]
            assert isinstance(results, Tensor)
            assert results.device == torch.device("cpu")
            assert results.dtype == torch.int32
            assert results.dim() == 2
            assert results.size(0) == _internal_batch_size
            assert results.size(1) == NUM_RESULTS

            end_of_round: Tensor = data[11]
            assert isinstance(end_of_round, Tensor)
            assert end_of_round.device == torch.device("cpu")
            assert end_of_round.dtype == torch.bool
            assert end_of_round.dim() == 1
            assert end_of_round.size(0) == _internal_batch_size

            done: Tensor = data[12]
            assert isinstance(done, Tensor)
            assert done.device == torch.device("cpu")
            assert done.dtype == torch.bool
            assert done.dim() == 1
            assert done.size(0) == _internal_batch_size

            td = TensorDict(
                {},
                batch_size=_internal_batch_size,
                device=torch.device("cpu"),
            )
            td.set("sparse", sparse)
            td.set("numeric", numeric)
            td.set("progression", progression)
            td.set("candidates", candidates)
            td.set("action", action)
            td.set(("next", "sparse"), next_sparse)
            td.set(("next", "numeric"), next_numeric)
            td.set(("next", "progression"), next_progression)
            td.set(("next", "candidates"), next_candidates)
            td.set(("next", "round_summary"), round_summary)
            td.set(("next", "results"), results)
            td.set(("next", "end_of_round"), end_of_round)
            td.set(("next", "end_of_game"), done.detach().clone())
            td.set(("next", "done"), done)

            annotations = torch.cat((annotations, td))  # type: ignore

            while True:
                assert isinstance(annotations, TensorDictBase)

                i = 0
                while i < annotations.batch_size[0]:
                    _done: Tensor = annotations["next", "done"][i]
                    assert isinstance(_done, Tensor)
                    if _done.item():
                        break
                    i += 1
                if i == annotations.batch_size[0]:
                    break

                length = i + 1
                assert torch.all(
                    ~annotations["next", "done"][: length - 1]
                ).item()
                assert annotations["next", "done"][length - 1].item()

                episode: TensorDict = annotations[:length].to_tensordict()
                assert isinstance(episode, TensorDict)
                with torch.no_grad():
                    self.__get_reward(episode, self.__contiguous)
                if episode.get(("next", "reward"), None) is None:
                    errmsg = (
                        "`get_reward` did not set the "
                        '`("next", "reward")` tensor.'
                    )
                    raise RuntimeError(errmsg)
                reward: Tensor = episode["next", "reward"]
                assert isinstance(reward, Tensor)
                if reward.dim() not in (1, 2):
                    errmsg = "An invalid shape of the `reward` tensor."
                    raise RuntimeError(errmsg)
                if reward.dim() == 2:
                    if reward.size(1) != 1:
                        errmsg = "An invalid shape of the `reward` tensor."
                        raise RuntimeError(errmsg)
                    reward.squeeze_(1)
                if reward.size(0) != length:
                    errmsg = "An invalid shape of the `reward` tensor."
                    raise RuntimeError(errmsg)
                if reward.dtype not in (
                    torch.float64,
                    torch.float32,
                    torch.float16,
                ):
                    errmsg = "An invalid `dtype` of the `reward` tensor."
                    raise RuntimeError(errmsg)
                episode["next", "reward"] = reward.to(self.__dtype)

                if progress is not None:
                    if self.__size + length <= self.__max_size:
                        progress.update(length)
                    else:
                        progress.update(self.__max_size - self.__size)

                # Check if the replay buffer would overflow.
                flag = False
                while True:
                    if self.__size + length > self.__max_size:
                        batch: TensorDict = (
                            self.__replay_buffer.sample().to_tensordict()
                        )
                        assert batch.size(0) == self.__batch_size * world_size
                        self.__size -= self.__batch_size * world_size
                        if world_size >= 2:
                            batch = batch[
                                self.__batch_size * rank : self.__batch_size
                                * (rank + 1)
                            ]
                        self.__batch_queue.append(batch)
                        flag = True
                        continue
                    break

                assert self.__size + length <= self.__max_size
                self.__replay_buffer.extend(episode)
                self.__size += length
                annotations = annotations[length:]

                if flag:
                    if progress is not None:
                        progress.close()
                    assert len(self.__batch_queue) >= 1
                    return self.__batch_queue.pop(0)

        raise RuntimeError("Never reach here.")
