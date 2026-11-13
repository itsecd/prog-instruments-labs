"""Загрузка JSON-конфига и публикация псевдонимов настроек
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Загрузить конфигурацию из JSON-файла."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


CONFIG: dict[str, Any] = load_config()


DEFAULT_ALGORITHM = CONFIG["hashing"]["default_algorithm"]
SUPPORTED_ALGORITHMS = tuple(CONFIG["hashing"]["supported_algorithms"])
TEXT_ENCODING = CONFIG["hashing"]["text_encoding"]
DECODE_ERRORS = CONFIG["hashing"]["decode_errors"]

DEFAULT_EXPERIMENT_COUNT = CONFIG["experiments"]["default_count"]
MIN_EXPERIMENT_COUNT = CONFIG["experiments"]["min_count"]
MAX_EXPERIMENT_COUNT = CONFIG["experiments"]["max_count"]
ALL_MODIFICATIONS = CONFIG["experiments"]["all_modifications_alias"]
MODIFICATION_TYPES = tuple(CONFIG["experiments"]["modification_types"])
MODIFICATION_LABELS = CONFIG["experiments"]["modification_labels"]

HEX_DIGIT_BITS = CONFIG["bits"]["hex_digit"]
BYTE_BITS = CONFIG["bits"]["byte"]

EXCELLENT_AVALANCHE_PERCENT = CONFIG["quality"]["excellent_percent"]
MODERATE_AVALANCHE_PERCENT = CONFIG["quality"]["moderate_percent"]
WARNING_AVALANCHE_PERCENT = CONFIG["quality"]["warning_percent"]
IDEAL_AVALANCHE_PERCENT = CONFIG["quality"]["ideal_percent"]
QUALITY_LEVELS = CONFIG["quality"]["levels"]

HASH_PREVIEW_LENGTH = CONFIG["display"]["hash_preview_length"]
TABLE_HASH_PREVIEW_LENGTH = CONFIG["display"]["table_hash_preview_length"]
TEXT_PREVIEW_LENGTH = CONFIG["display"]["text_preview_length"]
PROGRESS_BAR_LENGTH = CONFIG["display"]["progress_bar_length"]
PROGRESS_NCOLS = CONFIG["display"]["progress_ncols"]

GUI_WINDOW_TITLE = CONFIG["gui"]["window_title"]
GUI_MIN_WIDTH = CONFIG["gui"]["min_width"]
GUI_MIN_HEIGHT = CONFIG["gui"]["min_height"]
GUI_DEFAULT_TEXT = CONFIG["gui"]["default_text"]
GUI_SPLITTER_SIZES = CONFIG["gui"]["splitter_sizes"]
GUI_COLORS = CONFIG["gui"]["colors"]
