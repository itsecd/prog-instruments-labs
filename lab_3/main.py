import csv
import re

from checksum import calculate_checksum, serialize_result

PATTERNS = {
    "email": re.compile(r"^[\w.]+@(?:[\w-]+\.)+[\w-]+$"),
    "height": re.compile(r"^[0-2]\.\d{2}$"),
    "inn": re.compile(r"^\d{12}$"),
    "passport": re.compile(r"^\d{2} \d{2} \d{6}$"),
    "occupation": re.compile(r"^[a-zA-Zа-яА-ЯёЁ\s-]+$"),
    "latitude": re.compile(
        r"^[-+]?(?:(?:[0-8]?\d(?:\.\d+)?|90(?:\.0+)?)(?:[eE][-+]?\d+)?)$"
    ),
    "hex_color": re.compile(r"^#[0-9a-fA-F]{6}$"),
    "issn": re.compile(r"^\d{4}-\d{4}$"),
    "uuid": re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    ),
    "time": re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d{1,6})?$"),
}


def find_invalid_rows(csv_path: str) -> list[int]:
    invalid_rows: list[int] = []
    error_counts: dict[str, int] = {k: 0 for k in PATTERNS}

    with open(csv_path, "r", encoding="utf-16") as f:
        reader = csv.reader(f, delimiter=";")
        try:
            headers = next(reader)
        except StopIteration:
            return invalid_rows

        clean_headers = [h.strip('"').strip("\ufeff") for h in headers]

        for row_idx, row in enumerate(reader):
            row_has_error = False
            for col_idx, value in enumerate(row):
                val_clean = value.strip('"')
                col_name = clean_headers[col_idx]

                if col_name in PATTERNS and not PATTERNS[col_name].match(val_clean):
                        error_counts[col_name] += 1
                        row_has_error = True

            if row_has_error:
                invalid_rows.append(row_idx)

    print("Ошибки по колонкам:")
    for col, count in error_counts.items():
        print(f"  {col}: {count}")

    return invalid_rows


def main() -> None:
    csv_file = "lab_3/15.csv"
    variant = 15

    invalid_rows = find_invalid_rows(csv_file)
    print(f"Найдено невалидных строк: {len(invalid_rows)}")

    checksum = calculate_checksum(invalid_rows)
    print(f"Вычисленная контрольная сумма: {checksum}")

    serialize_result(variant, checksum)
    print("Результат успешно записан в lab_3/result.json")


if __name__ == "__main__":
    main()
