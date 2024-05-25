#!/usr/bin/env python3

from argparse import ArgumentParser
import sys


_RoundKey = tuple[str, int, int, int, int]


def _get_round_key(uuid: str, sparse: str, numeric: str) -> _RoundKey:
    sparse_fields = [int(x) for x in sparse.split(",")]
    numeric_fields = [int(x) for x in numeric.split(",")]

    seat = sparse_fields[6] - 71
    assert 0 <= seat and seat < 4
    chang = sparse_fields[7] - 75
    assert 0 <= chang and chang < 3
    ju = sparse_fields[8] - 78
    assert 0 <= ju and ju < 4

    ben = numeric_fields[0]
    assert ben >= 0

    return uuid, seat, chang, ju, ben


def _filter(room_filter: int | None) -> None:
    keys: set[_RoundKey] = set()

    for line in sys.stdin:
        line = line.rstrip("\n")

        columns = line.split("\t")
        if len(columns) != 8:
            raise RuntimeError(f"An invalid line: {line}")
        uuid, sparse, numeric, progression, _, _, _, _ = columns

        sparse_fields = [int(x) for x in sparse.split(",")]

        room = sparse_fields[0]
        if room_filter is not None:
            if room < room_filter:
                continue
        seat = sparse_fields[6] - 71

        progression_fields = [int(x) for x in progression.split(",")]

        bor = True
        for p in progression_fields:
            if p == 0:
                continue
            if 5 <= p and p <= 596:
                if (p - 5) // 148 == seat:
                    bor = False
                    break
                continue
            if 597 <= p and p <= 956:
                if (p - 597) // 90 == seat:
                    bor = False
                    break
                continue
            if 957 <= p and p <= 1436:
                if (p - 957) // 120 == seat:
                    bor = False
                    break
                continue
            if 1437 <= p and p <= 1880:
                if (p - 1437) // 111 == seat:
                    bor = False
                    break
                continue
            if 1881 <= p and p <= 2016:
                if (p - 1881) // 34 == seat:
                    bor = False
                    break
                continue
            if 2017 <= p and p <= 2164:
                if (p - 2017) // 37 == seat:
                    bor = False
                    break
                continue
            assert p == 2165
        if not bor:
            continue

        key = _get_round_key(uuid, sparse, numeric)
        if key in keys:
            continue

        keys.add(key)
        print(line)


def _main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--filter-by-room",
        choices=("bronze", "silver", "gold", "jade", "throne"),
        help="filter annotations by the specified room or above",
    )
    args = parser.parse_args()

    if args.filter_by_room is None:
        room_filter = None
    else:
        room_filter = {
            "bronze": 0,
            "silver": 1,
            "gold": 2,
            "jade": 3,
            "throne": 4,
        }[args.filter_by_room]

    _filter(room_filter)


if __name__ == "__main__":
    _main()
