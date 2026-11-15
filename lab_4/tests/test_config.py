"""Тесты конфигурации."""
from pathlib import Path

from src.config import DATA_PATH, RESULT_PATH, VARIANT


def test_data_path_exists():
    """DATA_PATH указывает на существующий файл."""
    assert DATA_PATH.exists(), f"файл не найден: {DATA_PATH}"
    assert DATA_PATH.is_file()


def test_data_path_is_csv():
    """Расширение — .csv."""
    assert DATA_PATH.suffix == ".csv"


def test_result_path_is_json():
    """result.json лежит в корне лабы."""
    assert RESULT_PATH.name == "result.json"
    assert RESULT_PATH.parent == Path(__file__).resolve().parent.parent


def test_variant_is_int():
    """Номер варианта — целое число."""
    assert isinstance(VARIANT, int)
    assert VARIANT > 0
