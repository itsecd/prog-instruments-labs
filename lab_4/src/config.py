# src/config.py
from pathlib import Path

# config.py лежит в src/, а data/ и result.json — в корне lab_4/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = _PROJECT_ROOT / "data" / "36.csv"
RESULT_PATH = _PROJECT_ROOT / "result.json"

VARIANT = 36
CSV_ENCODING = "utf-16"
CSV_DELIMITER = ";"
FIRST_DATA_ROW_INDEX = 0
