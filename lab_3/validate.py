import argparse
import csv
import json
from pathlib import Path

from checksum import calculate_checksum
from patterns import PATTERNS

VARIANT = 17
LAB_DIR = Path(__file__).resolve().parent


def invalid_rows(csv_path: Path) -> list[int]:
    with csv_path.open(encoding="utf-16", newline="") as file:
        reader = csv.reader(file, delimiter=";")
        header = next(reader, [])
        if len(header) != len(PATTERNS) or set(header) != PATTERNS.keys():
            raise ValueError("CSV header does not match variant 17")

        patterns = [PATTERNS[column] for column in header]
        return [
            index
            for index, row in enumerate(reader)
            if len(row) != len(patterns)
            or any(
                pattern.fullmatch(value) is None
                for pattern, value in zip(patterns, row)
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CSV for variant 17.")
    parser.add_argument("csv", type=Path, nargs="?", default=LAB_DIR / "17.csv")
    parser.add_argument("--output", type=Path, default=LAB_DIR / "result.json")
    args = parser.parse_args()

    try:
        rows = invalid_rows(args.csv)
        checksum = calculate_checksum(rows)
        result = {"variant": str(VARIANT), "checksum": checksum}
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"{error}\n")

    print(f"Invalid rows: {len(rows)}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()
