import json
import re

import pandas as pd
from checksum import calculate_checksum


def clean_utf16_string(s):
    """Очищает строку от нулевых байтов UTF-16."""
    if isinstance(s, str):
        s = s.replace('\x00', '')
        s = s.strip('"')
        return s
    return s


def validate_data(df: pd.DataFrame) -> list[int]:
    """
    Валидация данных CSV файла с использованием регулярных выражений.
    """
    invalid_rows = []

    patterns = {
        "email": re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        ),
        "height": re.compile(r'^[1-2]\.\d{2}$'),
        "snils": re.compile(r'^\d{11}$'),
        "passport": re.compile(r'^\d{2} \d{2} \d{6}$'),
        "occupation": re.compile(r'^[a-zA-Zа-яА-ЯёЁ\s\-–—]+$'),
        "longitude": re.compile(
            r'^-?(180(\.0+)?|1[0-7][0-9](\.\d+)?|[0-9]?[0-9](\.\d+)?)$'
        ),
        "hex_color": re.compile(r'^#[0-9a-fA-F]{6}$'),
        "issn": re.compile(r'^\d{4}-\d{4}$'),
        "locale_code": re.compile(r'^[a-z]{2}(-[a-z]{2,4})?$'),
        "time": re.compile(
            r'^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{1,6}$'
        )
    }

    for index, row in df.iterrows():
        is_valid = True

        for col, pattern in patterns.items():
            value = str(row[col])
            value = clean_utf16_string(value)

            if value == 'nan' or not value:
                is_valid = False
                break

            cleaned_value = value

            if col == "longitude":
                cleaned_value = re.sub(r'[°"]', '', cleaned_value)

            if not pattern.match(cleaned_value):
                is_valid = False
                break

        if not is_valid:
            invalid_rows.append(index)

    return invalid_rows


def strict_validate_data(df: pd.DataFrame) -> list[int]:
    """
    Строгая валидация - данные должны быть правильными без исправлений.
    """
    invalid_rows = []

    patterns = {
        "email": re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        ),
        "height": re.compile(r'^[1-2]\.\d{2}$'),
        "snils": re.compile(r'^\d{11}$'),
        "passport": re.compile(r'^\d{2} \d{2} \d{6}$'),
        "occupation": re.compile(r'^[a-zA-Zа-яА-ЯёЁ\s\-–—]+$'),
        "longitude": re.compile(
            r'^-?(180(\.0+)?|1[0-7][0-9](\.\d+)?|[0-9]?[0-9](\.\d+)?)$'
        ),
        "hex_color": re.compile(r'^#[0-9a-fA-F]{6}$'),
        "issn": re.compile(r'^\d{4}-\d{4}$'),
        "locale_code": re.compile(r'^[a-z]{2}(-[a-z]{2,4})?$'),
        "time": re.compile(
            r'^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{1,6}$'
        )
    }

    for index, row in df.iterrows():
        is_valid = True

        for col, pattern in patterns.items():
            value = str(row[col])
            value = clean_utf16_string(value)

            if value == 'nan' or not value:
                is_valid = False
                break

            if not pattern.match(value):
                is_valid = False
                break

        if not is_valid:
            invalid_rows.append(index)

    return invalid_rows


def main() -> None:
    """Основная функция выполнения лабораторной работы."""
    try:
        df = pd.read_csv(
            "19.csv",
            sep=";",
            encoding='utf-16',
            on_bad_lines='skip',
            header=0
        )
        print("Файл успешно загружен в кодировке UTF-16")
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return

    print(f"Размер данных: {df.shape}")

    for col in df.columns:
        df[col] = df[col].apply(clean_utf16_string)

    print("Запуск СТРОГОЙ валидации...")
    invalid_rows = strict_validate_data(df)

    checksum_value = calculate_checksum(invalid_rows)

    result_data = {
        "variant": 19,
        "checksum": checksum_value
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print("\n=== РЕЗУЛЬТАТЫ СТРОГОЙ ВАЛИДАЦИИ ===")
    print(f"Всего строк: {len(df)}")
    print(f"Невалидных строк: {len(invalid_rows)}")
    print(f"Валидных строк: {len(df) - len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum_value}")
    expected = "84f395ceeba40fb8d5799d91158e7175"
    print(f"Ожидаемая сумма: {expected}")
    print(f"Совпадает: {checksum_value == expected}")


if __name__ == "__main__":
    main()