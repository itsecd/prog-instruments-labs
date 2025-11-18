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
        "height": re.compile(r'^[1-2]\.\d{2}$'),  # 1.50 - 2.99
        "snils": re.compile(r'^\d{11}$'),  # 11 цифр
        "passport": re.compile(r'^\d{2} \d{2} \d{6}$'),  # 00 00 000000
        "occupation": re.compile(r'^[a-zA-Zа-яА-ЯёЁ\s\-–—]+$'),  # только буквы, пробелы, дефисы
        "longitude": re.compile(r'^-?(180(\.0+)?|1[0-7][0-9](\.\d+)?|[0-9]?[0-9](\.\d+)?)$'),  # -180.0 до 180.0
        "hex_color": re.compile(r'^#[0-9a-fA-F]{6}$'),  # #ffffff
        "issn": re.compile(r'^\d{4}-\d{4}$'),  # 0000-0000
        "locale_code": re.compile(r'^[a-z]{2}(-[A-Za-z]{2,4})?$'),  # en или en-US
        "time": re.compile(r'^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{1,6}$')  # время с микросекундами
    }

    for index, row in df.iterrows():
        is_valid = True

        for col, pattern in patterns.items():
            value = str(row[col])

            # Очистка UTF-16 артефактов
            value = clean_utf16_string(value)

            # Пропускаем NaN значения
            if value == 'nan' or not value:
                is_valid = False
                break

            # Специфичная очистка для каждого поля
            cleaned_value = value

            if col == "snils":
                cleaned_value = value.replace('_', '')
            elif col == "longitude":
                cleaned_value = re.sub(r'[°"_]', '', value)
                cleaned_value = cleaned_value.replace(',', '.')
                # Убедимся, что это число в допустимом диапазоне
                try:
                    lon = float(cleaned_value)
                    if lon < -180 or lon > 180:
                        is_valid = False
                        break
                except ValueError:
                    is_valid = False
                    break
            elif col == "issn":
                cleaned_value = value.replace(' ', '')
            elif col == "locale_code":
                cleaned_value = value.strip('_')
            elif col == "hex_color":
                if value.startswith('##'):
                    cleaned_value = value[1:]
                cleaned_value = cleaned_value.upper()

            # Проверка по регулярному выражению
            if not pattern.match(cleaned_value):
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


def debug_validation(df: pd.DataFrame) -> None:
    """Функция для отладки валидации"""
    print("\n=== ОТЛАДКА ВАЛИДАЦИИ ===")

    patterns = {
        "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        "height": re.compile(r'^[1-2]\.\d{2}$'),
        "snils": re.compile(r'^\d{11}$'),
        "passport": re.compile(r'^\d{2} \d{2} \d{6}$'),
        "occupation": re.compile(r'^[a-zA-Zа-яА-ЯёЁ\s\-–—]+$'),
        "longitude": re.compile(r'^-?(180(\.0+)?|1[0-7][0-9](\.\d+)?|[0-9]?[0-9](\.\d+)?)$'),
        "hex_color": re.compile(r'^#[0-9a-fA-F]{6}$'),
        "issn": re.compile(r'^\d{4}-\d{4}$'),
        "locale_code": re.compile(r'^[a-z]{2}(-[A-Za-z]{2,4})?$'),
        "time": re.compile(r'^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d\.\d{1,6}$')
    }

    # Проверим первые 5 строк подробно
    for i in range(min(5, len(df))):
        print(f"\n--- Строка {i} ---")
        for col, pattern in patterns.items():
            value = str(df.iloc[i][col])
            cleaned_value = clean_utf16_string(value)

            # Специфичная очистка
            if col == "snils":
                cleaned_value = cleaned_value.replace('_', '')
            elif col == "longitude":
                cleaned_value = re.sub(r'[°"_]', '', cleaned_value)
                cleaned_value = cleaned_value.replace(',', '.')
            elif col == "issn":
                cleaned_value = cleaned_value.replace(' ', '')
            elif col == "locale_code":
                cleaned_value = cleaned_value.strip('_')
            elif col == "hex_color":
                if cleaned_value.startswith('##'):
                    cleaned_value = cleaned_value[1:]
                cleaned_value = cleaned_value.upper()

            matches = bool(pattern.match(cleaned_value))
            print(f"{col}: '{value}' -> '{cleaned_value}' | {'VALID' if matches else 'INVALID'}")


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
    print(f"Столбцы: {df.columns.tolist()}")

    # Очистка данных от UTF-16 артефактов
    for col in df.columns:
        df[col] = df[col].apply(clean_utf16_string)

    # Отладка первых строк
    debug_validation(df)

    # Валидация всех данных
    invalid_rows = validate_data(df)
    checksum_value = calculate_checksum(invalid_rows)

    save_to_json(19, checksum_value)

    print("\n=== РЕЗУЛЬТАТЫ ВАЛИДАЦИИ ===")
    print(f"Вариант: 19")
    print(f"Всего строк: {len(df)}")
    print(f"Невалидных строк: {len(invalid_rows)}")
    print(f"Валидных строк: {len(df) - len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum_value}")
    print(f"Ожидаемая сумма: 84f395ceeba40fb8d5799d91158e7175")


if __name__ == "__main__":
    main()