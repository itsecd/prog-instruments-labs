from files_handler import get_csv_path, load_csv
from validator import validate_row
from lab_3.checksum import calculate_checksum, serialize_result

VARIANT = 6

def main():
    csv_path = get_csv_path()
    df = load_csv(csv_path)

    invalid_rows = []

    for idx, row in df.iterrows():
        if not validate_row(row):
            invalid_rows.append(idx - 1)

    checksum = calculate_checksum(invalid_rows)

    serialize_result(VARIANT, checksum)

    print(f"Общее количество строк: {len(df)}")
    print(f"Найдено невалидных строк: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum}")


if __name__ == "__main__":
    main()