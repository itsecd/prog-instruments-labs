import pandas as pd
import re
import json
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

    Args:
        df: DataFrame с данными для валидации

    Returns:
        list[int]: Список индексов невалидных строк
    """
    invalid_rows = []

    patterns = {
        "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        "height": re.compile(r'^[1-2]\.\d{2}$'),
        "snils": re.compile(r'^\d{11}$'),
        "passport": re.compile(r'^\d{2} \d{2} \d{6}$'),
        "occupation": re.compile(r'^[а-яА-Яa-zA-Z\s\-\d_]+$'),
        "longitude": re.compile(r'^-?\d{1,3}\.\d+$'),
        "hex_color": re.compile(r'^#[0-9a-fA-F]{6}$'),
        "issn": re.compile(r'^\d{4}-\d{4}$'),
        "locale_code": re.compile(r'^[a-z]{2}(-[a-z]{2,4})?$'),
        "time": re.compile(r'^\d{2}:\d{2}:\d{2}\.\d+$')
    }

    for index, row in df.iterrows():
        is_valid = True

        for col, pattern in patterns.items():
            value = str(row[col])

            value = clean_utf16_string(value)

            if not value or value == 'nan':
                is_valid = False
                break

            if col == "snils":
                value = value.strip('_')
            elif col == "longitude":
                value = re.sub(r'[°_"]', '', value)
                value = value.replace(',', '.')
            elif col == "issn":
                value = value.replace(' ', '')
            elif col == "locale_code":
                value = value.strip('_')
            elif col == "hex_color":
                if value.startswith('##'):
                    value = value[1:]
                if len(value) > 7:
                    value = value[:7]

            if not pattern.match(value):
                is_valid = False
                break

        if not is_valid:
            invalid_rows.append(index)

    return invalid_rows


def save_to_json(variant: int, checksum: str) -> None:
    """
    Сохранение результата в JSON файл.

    Args:
        variant: Номер варианта
        checksum: Контрольная сумма
    """
    result_data = {
        "variant": variant,
        "checksum": checksum
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print("Результат записан в result.json")

    try:
        with open("result.json", "r", encoding="utf-8") as f:
            content = f.read()
        print("Файл result.json создан успешно")
        print(f"Содержимое: {content}")
    except Exception as e:
        print(f"Ошибка при проверке файла: {e}")


def main() -> None:
    """Основная функция выполнения лабораторной работы."""
    try:
        df = pd.read_csv(
            "19.csv",
            sep=";",
            encoding='utf-16',
            on_bad_lines='skip',
            header=0,
            skiprows=[1]
        )

        print("Файл успешно загружен в кодировке UTF-16")

    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return

    print(f"Размер данных: {df.shape}")
    print(f"Столбцы: {df.columns.tolist()}")

    for col in df.columns:
        df[col] = df[col].apply(clean_utf16_string)

    invalid_rows = validate_data(df)
    checksum_value = calculate_checksum(invalid_rows)

    save_to_json(19, checksum_value)

    print("\n=== РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ===")
    print(f"Вариант: 19")
    print(f"Всего строк: {len(df)}")
    print(f"Невалидных строк: {len(invalid_rows)}")
    print(f"Валидных строк: {len(df) - len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum_value}")


if __name__ == "__main__":
    main()