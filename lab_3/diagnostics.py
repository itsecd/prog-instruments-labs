import csv
from collections import Counter
from src.validators import VALIDATORS


def count_per_field(filepath: str) -> None:
    counter = Counter()
    examples = {}  # field -> 5 примеров невалидных значений

    with open(filepath, encoding="utf-16", newline="") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            for field, value in row.items():
                pattern = VALIDATORS.get(field)
                if pattern is None:
                    print(f"⚠️  НЕТ ВАЛИДАТОРА для поля {field!r}")
                    continue
                if not pattern.fullmatch((value or "").strip()):
                    counter[field] += 1
                    examples.setdefault(field, []).append(value)

    print(f"\n{'поле':25} {'невалидных':>12}")
    print("-" * 40)
    for field, n in counter.most_common():
        print(f"{field:25} {n:>12}")
        # показываем примеры «битых» значений
        for v in examples[field][:5]:
            print(f"    {v!r}")
    print("-" * 40)
    print(f"{'ИТОГО':25} {sum(counter.values()):>12}")


if __name__ == "__main__":
    count_per_field("data/36.csv")
