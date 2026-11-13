"""Настройка журнала для консольного и графического интерфейсов."""

import logging
import os
import sys
import time
from pathlib import Path

LOG_PATH = Path(__file__).with_name("avalanche.log")
LOGGER_NAME = "avalanche"
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def configure_logging() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return

    requested_level = os.getenv("AVALANCHE_LOG_LEVEL", "INFO").upper()
    valid_level = requested_level in LEVELS
    logger.setLevel(requested_level if valid_level else "INFO")
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)s "
        "%(name)s.%(funcName)s:%(lineno)d [pid=%(process)d] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime

    file_unavailable = False
    try:
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    except OSError:
        handler = logging.StreamHandler(sys.stderr)
        file_unavailable = True

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if file_unavailable:
        logger.warning("Could not open avalanche.log. Writing logs to stderr instead")

    if not valid_level:
        logger.warning("Invalid AVALANCHE_LOG_LEVEL. Using INFO")
