import json
import hashlib
import csv
import re
import chardet

from typing import Dict, List


patterns: Dict[str, str] = {
    "email": r"^[a-z0-9]+(?:[._][a-z0-9]+)*\@[a-z]+(?:\.[a-z]+)+$",
    "height": r"^[1-2]\.\d{2}$",
    "inn": r"^\d{12}$",
    "passport": r"^\d{2}\s\d{2}\s\d{6}$",
    "occupation": r"[a-zA-Zа-яА-ЯёЁ -]+",
    "latitude": r"^-?(90|[0-8]?[0-9])\.\d+$",
    "hex_color": r"^\#[0-9a-fA-F]{6}$",
    "issn": r"^\d{4}-\d{4}$",
    "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
    "time": r"^(2[0-3]|[0-1][0-9]):[0-5][0-9]:[0-5][0-9]\.\d{6}$"
}


def check_patterns(row: List[str], row_number: int) -> bool:
    """
    Проверяет строку на соответствие заданным паттернам.

    :param row: Список значений строки.
    :param row_number: Номер строки в CSV-файле.
    :return: True, если найдена ошибка, иначе False.
    """
    for key, value in zip(patterns.keys(), row):
        if not re.match(patterns[key], value):
            print(f"Ошибка в строке {row_number}: {value} не соответствует паттерну {key}.")
            return False
    return True


def detect_encoding(file_path: str) -> str:
    """
    Определяет кодировку файла.

    :param file_path: Путь к файлу.
    :return: Кодировка файла.
    :raises RuntimeError: Если произошла ошибка при определении кодировки.
    """
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']
    except Exception as e:
        raise RuntimeError(f"Ошибка при определении кодировки файла: {e}")


def process_csv(file_path: str) -> List[int]:
    """
    Обрабатывает CSV-файл и возвращает номера строк с ошибками.

    :param file_path: Путь к CSV-файлу.
    :return: Список номеров строк с ошибками.
    :raises FileNotFoundError: Если файл не найден.
    :raises RuntimeError: Если произошла ошибка при обработке CSV файла.
    """
    error_rows: List[int] = []
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, newline='', encoding=encoding) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            next(reader)  # Пропустить заголовок

            for row_number, row in enumerate(reader, start=1):
                if not check_patterns(list(row.values()), row_number):
                    error_rows.append(row_number)

    except FileNotFoundError:
        print(f"Файл не найден: {file_path}")
        raise
    except Exception as e:
        raise RuntimeError(f"Ошибка при обработке CSV файла: {e}")

    return error_rows


def calculate_checksum(row_numbers: List[int]) -> str:
    """
    Вычисляет контрольную сумму для списка номеров строк.

    :param row_numbers: Список номеров строк.
    :return: Контрольная сумма в формате MD5.
    """
    row_numbers.sort()
    return hashlib.md5(json.dumps(row_numbers).encode('utf-8')).hexdigest()


def serialize_to_json(variant: int, checksum: str) -> None:
    """
    Сериализует результат в JSON-файл.

    :param variant: Номер варианта.
    :param checksum: Контрольная сумма.
    :raises RuntimeError: Если произошла ошибка при записи в файл.
    """
    try:
        with open('lab_3/result.json', 'w', encoding='utf-8') as file:
            result = {
                "variant": variant,
                "checksum": checksum
            }
            json.dump(result, file, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Ошибка при записи в файл: {e}")


if __name__ == "__main__":
    try:
        with open("lab_3/path_csv.json", "r", encoding='utf-8') as options_file:
            options = json.load(options_file)

        error_data = process_csv(options["csv_file_path"])
        checksum = calculate_checksum(error_data)
        variant = 39
        serialize_to_json(variant, checksum)
    except Exception as e:
        print(f"Произошла ошибка: {e}")
