"""Диагностика валидации: количество невалидных значений по полям."""
import argparse
from collections import Counter

from config import DATA_PATH
from reader import read_csv
from validators import find_invalid_fields


def count_per_field(filepath, examples_limit: int = 5) -> Counter:
    """Считает невалидные значения по полям и печатает примеры."""
    counter = Counter()
    examples: dict[str, list[str]] = {}

    for row in read_csv(filepath):
        for field in find_invalid_fields(row):
            counter[field] += 1
            examples.setdefault(field, []).append(row.get(field, ""))

    print(f"\n{'поле':25} {'невалидных':>12}")
    print("-" * 40)
    for field, n in counter.most_common():
        print(f"{field:25} {n:>12}")
        for value in examples[field][:examples_limit]:
            print(f"    {value!r}")
    print("-" * 40)
    print(f"{'ИТОГО':25} {sum(counter.values()):>12}")
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Диагностика валидации CSV")
    parser.add_argument("--data", default=DATA_PATH, help="путь к CSV")
    args = parser.parse_args()
    count_per_field(args.data)


if __name__ == "__main__":
    main()
