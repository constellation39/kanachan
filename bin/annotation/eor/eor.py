#!/usr/bin/env python3

from argparse import ArgumentParser
import sys


def _filter(room_filter: int | None) -> None:
    round2annotation = {}
    for line in sys.stdin:
        line = line.rstrip("\n")

        columns = line.split("\t")
        if len(columns) != 8:
            raise RuntimeError(f"An invalid line: {line}")
        uuid, sparse, numeric, progression, _, _, _, results = columns

        sparse_fields = [int(x) for x in sparse.split(",")]

        room = sparse_fields[0]
        if room_filter is not None:
            if room < room_filter:
                continue
        game_style = sparse_fields[1] - 5
        grade0 = sparse_fields[2] - 7
        grade1 = sparse_fields[3] - 23
        grade2 = sparse_fields[4] - 39
        grade3 = sparse_fields[5] - 55
        chang = sparse_fields[7] - 75
        ju = sparse_fields[8] - 78

        numeric_fields = [int(x) for x in numeric.split(",")]
        benchang = numeric_fields[0]

        progression_fields = [int(x) for x in progression.split(",")]
        if len(progression_fields) >= 2:
            continue

        result_fields = [int(x) for x in results.split(",")]

        round_sparse = (
            f"{room},{game_style + 5},{grade0 + 7},{grade1 + 23},{grade2 + 39}"
            f",{grade3 + 55},{chang + 71},{ju + 74}"
        )
        next_numeric = (
            f"{result_fields[4]},{result_fields[5]},{result_fields[6]}"
            f",{result_fields[7]},{result_fields[8]},{result_fields[9]}"
        )
        game_result = (
            f"{result_fields[10]},{result_fields[11]},{result_fields[12]}"
            f",{result_fields[13]}"
        )
        annotation = f"{uuid}\t{round_sparse}\t{next_numeric}\t{game_result}"

        round_key = (uuid, chang, ju, benchang)
        if round_key in round2annotation:
            raise RuntimeError("A logic error.")
        round2annotation[round_key] = annotation

    annotations = [(k, v) for k, v in round2annotation.items()]
    annotations.sort(key=lambda x: x[0])

    i = 0
    while i + 1 < len(annotations):
        prev_uuid, _, _, _ = annotations[i][0]
        next_uuid, _, _, _ = annotations[i + 1][0]
        if next_uuid != prev_uuid:
            i += 1
            continue

        print(annotations[i][1])

        i += 1


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
