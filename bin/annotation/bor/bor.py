#!/usr/bin/env python3

from argparse import ArgumentParser
import sys


def _filter(room_filter: int | None) -> None:
    for line in sys.stdin:
        line = line.rstrip("\n")

        columns = line.split("\t")
        if len(columns) != 8:
            raise RuntimeError(f"An invalid line: {line}")
        _, sparse, _, progression, _, _, _, _ = columns

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
            if (p - 5) // 148 == seat:
                bor = False
                break
        if not bor:
            continue

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
