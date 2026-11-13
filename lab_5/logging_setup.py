"""Logging configuration loader for the task tracker CLI."""

import json
import logging
import logging.config
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("logging_config.json")


def setup_logging(config_path: Path = CONFIG_PATH) -> None:
    """Configure logging from a JSON file, falling back to basicConfig."""
    try:
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        logging.config.dictConfig(config)
    except (OSError, ValueError) as exc:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("task_cli").warning(
            "Failed to load logging config %s: %s. Using basicConfig.",
            config_path,
            exc,
        )