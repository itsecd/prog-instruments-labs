"""Конфигурация лабораторной работы."""
from pathlib import Path

# Путь к CSV относительно корня лабы
DATA_PATH = Path(__file__).resolve().parent / "data" / "36.csv"

# Путь к result.json относительно корня лабы
RESULT_PATH = Path(__file__).resolve().parent / "result.json"

# Номер варианта (см. таблицу успеваемости)
VARIANT = 36

# Кодировка и разделитель CSV (файл — «Excel-подобный» CSV)
CSV_ENCODING = "utf-16"
CSV_DELIMITER = ";"

# Нумерация строк для checksum: первая строка данных имеет номер 0
# (см. докстринг calculate_checksum в checksum.py)
FIRST_DATA_ROW_INDEX = 0
