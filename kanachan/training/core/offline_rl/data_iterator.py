from pathlib import Path
import gzip
import bz2

from tqdm import tqdm
import torch
from torch.utils.data import get_worker_info
from kanachan.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES,
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    NUM_TYPES_OF_PROGRESSION_FEATURES,
    MAX_LENGTH_OF_PROGRESSION_FEATURES,
    NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES,
    NUM_TYPES_OF_ROUND_SUMMARY,
    MAX_NUM_ROUND_SUMMARY,
    NUM_RESULTS,
)


class DataIterator:
    def __init__(
        self,
        *,
        path: Path,
        num_skip_samples: int,
        rewrite_rooms: int | None,
        rewrite_grades: int | None,
        local_rank: int,
    ) -> None:
        if num_skip_samples < 0:
            errmsg = (
                f"{num_skip_samples}: An invalid value for `num_skip_samples`."
            )
            raise ValueError(errmsg)
        if rewrite_rooms is not None and (
            rewrite_rooms < 0 or 4 < rewrite_rooms
        ):
            errmsg = f"{rewrite_rooms}: An invalid value for `rewrite_rooms`."
            raise ValueError(errmsg)
        if rewrite_grades is not None and (
            rewrite_grades < 0 or 15 < rewrite_grades
        ):
            errmsg = (
                f"{rewrite_grades}: An invalid value for `rewrite_grades`."
            )
            raise ValueError(errmsg)

        if path.suffix == ".gz":
            self.__fp = gzip.open(path, mode="rt", encoding="UTF-8")
        elif path.suffix == ".bz2":
            self.__fp = bz2.open(path, mode="rt", encoding="UTF-8")
        else:
            self.__fp = open(path, encoding="UTF-8")

        self.__rewrite_rooms = rewrite_rooms
        self.__rewrite_grades = rewrite_grades

        worker_info = get_worker_info()

        if num_skip_samples > 0:
            is_primary_worker = worker_info is None or worker_info.id == 0
            with tqdm(
                desc="Skipping leading samples...",
                total=num_skip_samples,
                maxinterval=0.1,
                disable=(local_rank != 0 or not is_primary_worker),
                unit=" lines",
                smoothing=0.0,
            ) as progress:
                for _ in range(num_skip_samples):
                    self.__fp.readline()
                    progress.update()

        if worker_info is not None:
            try:
                assert worker_info is not None
                for _ in range(worker_info.id):
                    next(self.__fp)
            except StopIteration as _:
                pass

    def __del__(self) -> None:
        self.__fp.close()

    def __parse_line(self, line: str):
        line = line.rstrip("\n")
        columns = line.split("\t")
        if len(columns) not in (8, 10, 12):
            errmsg = f"An invalid line: {line}"
            raise RuntimeError(errmsg)

        _, sparse, numeric, progression, candidates, action = columns[:6]

        sparse = [int(x) for x in sparse.split(",")]
        if len(sparse) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
            errmsg = f"{len(sparse)} > {MAX_NUM_ACTIVE_SPARSE_FEATURES}"
            raise RuntimeError(errmsg)
        for x in sparse:
            if x >= NUM_TYPES_OF_SPARSE_FEATURES:
                errmsg = f"{x} >= {NUM_TYPES_OF_SPARSE_FEATURES}"
                raise RuntimeError(errmsg)
        for _ in range(len(sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
            # padding
            sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
        sparse = torch.tensor(
            sparse, device=torch.device("cpu"), dtype=torch.int32
        )
        if self.__rewrite_rooms is not None:
            sparse[0] = self.__rewrite_rooms
        if self.__rewrite_grades is not None:
            sparse[2] = 7 + self.__rewrite_grades
            sparse[3] = 23 + self.__rewrite_grades
            sparse[4] = 39 + self.__rewrite_grades
            sparse[5] = 55 + self.__rewrite_grades

        numeric = [int(x) for x in numeric.split(",")]
        if len(numeric) != NUM_NUMERIC_FEATURES:
            errmsg = f"{len(numeric)} != {NUM_NUMERIC_FEATURES}"
            raise RuntimeError(errmsg)
        numeric = torch.tensor(
            numeric, device=torch.device("cpu"), dtype=torch.int32
        )

        progression = [int(x) for x in progression.split(",")]
        if len(progression) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
            errmsg = (
                f"{len(progression)}"
                f" > {MAX_LENGTH_OF_PROGRESSION_FEATURES}"
            )
            raise RuntimeError(errmsg)
        for x in progression:
            if x >= NUM_TYPES_OF_PROGRESSION_FEATURES:
                errmsg = f"{x} >= {NUM_TYPES_OF_PROGRESSION_FEATURES}"
                raise RuntimeError(errmsg)
        for _ in range(len(progression), MAX_LENGTH_OF_PROGRESSION_FEATURES):
            # padding
            progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
        progression = torch.tensor(
            progression, device=torch.device("cpu"), dtype=torch.int32
        )

        candidates = [int(x) for x in candidates.split(",")]
        if len(candidates) > MAX_NUM_ACTION_CANDIDATES:
            errmsg = f"{len(candidates)} >= {MAX_NUM_ACTION_CANDIDATES}"
            raise RuntimeError(errmsg)
        for x in candidates:
            if x >= NUM_TYPES_OF_ACTIONS:
                errmsg = f"{x} >= {NUM_TYPES_OF_ACTIONS}"
                raise RuntimeError(errmsg)
        for _ in range(len(candidates), MAX_NUM_ACTION_CANDIDATES):
            # padding
            candidates.append(NUM_TYPES_OF_ACTIONS)
        candidates = torch.tensor(
            candidates, device=torch.device("cpu"), dtype=torch.int32
        )

        action = int(action)
        action = torch.tensor(
            action, device=torch.device("cpu"), dtype=torch.int32
        )

        if len(columns) in (10, 12):
            # Not end-of-game.
            (
                next_sparse,
                next_numeric,
                next_progression,
                next_candidates,
            ) = columns[6:10]

            next_sparse = [int(x) for x in next_sparse.split(",")]
            if len(next_sparse) > MAX_NUM_ACTIVE_SPARSE_FEATURES:
                errmsg = (
                    f"{len(next_sparse)}"
                    f" > {MAX_NUM_ACTIVE_SPARSE_FEATURES}"
                )
                raise RuntimeError(errmsg)
            for x in next_sparse:
                if x >= NUM_TYPES_OF_SPARSE_FEATURES:
                    errmsg = f"{x} >= {NUM_TYPES_OF_SPARSE_FEATURES}"
                    raise RuntimeError(errmsg)
            for _ in range(len(next_sparse), MAX_NUM_ACTIVE_SPARSE_FEATURES):
                # padding
                next_sparse.append(NUM_TYPES_OF_SPARSE_FEATURES)
            next_sparse = torch.tensor(
                next_sparse, device=torch.device("cpu"), dtype=torch.int32
            )
            if self.__rewrite_rooms is not None:
                next_sparse[0] = self.__rewrite_rooms
            if self.__rewrite_grades is not None:
                next_sparse[2] = 7 + self.__rewrite_grades
                next_sparse[3] = 23 + self.__rewrite_grades
                next_sparse[4] = 39 + self.__rewrite_grades
                next_sparse[5] = 55 + self.__rewrite_grades

            next_numeric = [int(x) for x in next_numeric.split(",")]
            if len(next_numeric) != NUM_NUMERIC_FEATURES:
                errmsg = f"{len(next_numeric)} != {NUM_NUMERIC_FEATURES}"
                raise RuntimeError(errmsg)
            next_numeric = torch.tensor(
                next_numeric, device=torch.device("cpu"), dtype=torch.int32
            )

            next_progression = [int(x) for x in next_progression.split(",")]
            if len(next_progression) > MAX_LENGTH_OF_PROGRESSION_FEATURES:
                errmsg = (
                    f"{len(next_progression)}"
                    f" > {MAX_LENGTH_OF_PROGRESSION_FEATURES}"
                )
                raise RuntimeError(errmsg)
            for x in next_progression:
                if x >= NUM_TYPES_OF_PROGRESSION_FEATURES:
                    errmsg = f"{x} >= {NUM_TYPES_OF_PROGRESSION_FEATURES}"
                    raise RuntimeError(errmsg)
            for _ in range(
                len(next_progression), MAX_LENGTH_OF_PROGRESSION_FEATURES
            ):
                # padding
                next_progression.append(NUM_TYPES_OF_PROGRESSION_FEATURES)
            next_progression = torch.tensor(
                next_progression, device=torch.device("cpu"), dtype=torch.int32
            )

            next_candidates = [int(x) for x in next_candidates.split(",")]
            if len(next_candidates) > MAX_NUM_ACTION_CANDIDATES:
                errmsg = (
                    f"{len(next_candidates)}" f" > {MAX_NUM_ACTION_CANDIDATES}"
                )
                raise RuntimeError(errmsg)
            for x in next_candidates:
                if x >= NUM_TYPES_OF_ACTIONS:
                    errmsg = f"{x} >= {NUM_TYPES_OF_ACTIONS}"
                    raise RuntimeError(errmsg)
            for _ in range(len(next_candidates), MAX_NUM_ACTION_CANDIDATES):
                # padding
                next_candidates.append(NUM_TYPES_OF_ACTIONS)
            next_candidates = torch.tensor(
                next_candidates, device=torch.device("cpu"), dtype=torch.int32
            )

            if len(columns) == 10:
                # Not end-of-game nor end-of-round.
                round_summary = torch.full(
                    (MAX_NUM_ROUND_SUMMARY,),
                    NUM_TYPES_OF_ROUND_SUMMARY,
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                )
                results = torch.zeros(
                    NUM_RESULTS,
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                )
                end_of_round = torch.tensor(
                    False, device=torch.device("cpu"), dtype=torch.bool
                )
            else:
                # End-of-round but not End-of-game
                assert len(columns) == 12
                round_summary, results = columns[10:]

                round_summary = [int(x) for x in round_summary.split(",")]
                if len(round_summary) == 0:
                    errmsg = f"An invalid line: {line}"
                    raise RuntimeError(errmsg)
                if len(round_summary) > MAX_NUM_ROUND_SUMMARY:
                    errmsg = (
                        f"{len(round_summary)}" f" > {MAX_NUM_ROUND_SUMMARY}"
                    )
                    raise RuntimeError(errmsg)
                for x in round_summary:
                    if x >= NUM_TYPES_OF_ROUND_SUMMARY:
                        errmsg = f"{x} >= {NUM_TYPES_OF_ROUND_SUMMARY}"
                        raise RuntimeError(errmsg)
                for _ in range(len(round_summary), MAX_NUM_ROUND_SUMMARY):
                    # Padding
                    round_summary.append(NUM_TYPES_OF_ROUND_SUMMARY)
                round_summary = torch.tensor(
                    round_summary,
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                )

                results = [int(x) for x in results.split(",")]
                if len(results) != NUM_RESULTS - 4:
                    errmsg = f"{len(results)} != {NUM_RESULTS - 4}"
                    raise RuntimeError(errmsg)
                results.extend([0, 0, 0, 0])
                results = torch.tensor(
                    results, device=torch.device("cpu"), dtype=torch.int32
                )

                end_of_round = torch.tensor(
                    True, device=torch.device("cpu"), dtype=torch.bool
                )

            done = torch.tensor(
                False, device=torch.device("cpu"), dtype=torch.bool
            )

            return (
                sparse,
                numeric,
                progression,
                candidates,
                action,
                next_sparse,
                next_numeric,
                next_progression,
                next_candidates,
                round_summary,
                results,
                end_of_round,
                done,
            )

        # End-of-game
        assert len(columns) == 8

        dummy_sparse = [
            NUM_TYPES_OF_SPARSE_FEATURES
        ] * MAX_NUM_ACTIVE_SPARSE_FEATURES
        dummy_sparse = torch.tensor(
            dummy_sparse, device=torch.device("cpu"), dtype=torch.int32
        )

        dummy_numeric = torch.zeros(
            NUM_NUMERIC_FEATURES, device=torch.device("cpu"), dtype=torch.int32
        )

        dummy_progression = [
            NUM_TYPES_OF_PROGRESSION_FEATURES
        ] * MAX_LENGTH_OF_PROGRESSION_FEATURES
        dummy_progression = torch.tensor(
            dummy_progression, device=torch.device("cpu"), dtype=torch.int32
        )

        dummy_candidates = [NUM_TYPES_OF_ACTIONS] * MAX_NUM_ACTION_CANDIDATES
        dummy_candidates = torch.tensor(
            dummy_candidates, device=torch.device("cpu"), dtype=torch.int32
        )

        round_summary, results = columns[6:]

        round_summary = [int(x) for x in round_summary.split(",")]
        if len(round_summary) == 0:
            errmsg = f"An invalid line: {line}"
            raise RuntimeError(errmsg)
        if len(round_summary) > MAX_NUM_ROUND_SUMMARY:
            errmsg = f"{len(round_summary)} > {MAX_NUM_ROUND_SUMMARY}"
            raise RuntimeError(errmsg)
        for x in round_summary:
            if x >= NUM_TYPES_OF_ROUND_SUMMARY:
                errmsg = f"{x} >= {NUM_TYPES_OF_ROUND_SUMMARY}"
                raise RuntimeError(errmsg)
        for _ in range(len(round_summary), MAX_NUM_ROUND_SUMMARY):
            # Padding
            round_summary.append(NUM_TYPES_OF_ROUND_SUMMARY)
        round_summary = torch.tensor(
            round_summary, device=torch.device("cpu"), dtype=torch.int32
        )

        results = [int(x) for x in results.split(",")]
        if len(results) != NUM_RESULTS:
            errmsg = f"{len(results)} != {NUM_RESULTS}"
            raise RuntimeError(errmsg)
        results = torch.tensor(
            results, device=torch.device("cpu"), dtype=torch.int32
        )

        end_of_round = torch.tensor(
            True, device=torch.device("cpu"), dtype=torch.bool
        )

        done = torch.tensor(True, device=torch.device("cpu"), dtype=torch.bool)

        return (
            sparse,
            numeric,
            progression,
            candidates,
            action,
            dummy_sparse,
            dummy_numeric,
            dummy_progression,
            dummy_candidates,
            round_summary,
            results,
            end_of_round,
            done,
        )

    def __next__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            line = next(self.__fp)
            return self.__parse_line(line)
        else:
            assert worker_info.num_workers >= 1
            line = next(self.__fp)
            try:
                for _ in range(worker_info.num_workers - 1):
                    next(self.__fp)
            except StopIteration:
                pass
            return self.__parse_line(line)
