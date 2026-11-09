"""Logging configuration: console + rotating file."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Configure root logger with console and rotating file handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Avoid duplicate handlers on repeated calls (useful in tests).
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level.upper())
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
