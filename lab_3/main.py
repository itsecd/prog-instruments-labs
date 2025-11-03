from file_io import read_csv, read_json, write_json
from validate import find_error_rows
from checksum import calculate_checksum

def main() -> None:
    """
    Читает CSV-файл и файл с регулярными выражениями, выполняет валидацию строк,
    вычисляет контрольную сумму и сохраняет результат в result.json.
    """
    patterns = read_json('patterns.json')
    rows = read_csv('50.csv')

    error_rows = find_error_rows(rows, patterns)

    checksum = calculate_checksum(error_rows)

    result = {"variant": 50, "checksum": checksum}
    write_json(result, 'result.json')

    print(f"Всего строк: {len(rows)}")
    print(f"Ошибок: {len(error_rows)}")
    print(f"Контрольная сумма: {checksum}")

if __name__ == "__main__":
    main()