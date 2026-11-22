import logging
import sys
from handler import SQLiteHandler


def setup_logger(db_path: str = "logs.db") -> logging.Logger:
    """
    Настраивает логгер: пишет в консоль (INFO+) и в SQLite (DEBUG+).
    """
    logger = logging.getLogger("log_parser")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    db_handler = SQLiteHandler(db_path)
    db_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(db_handler)

    return logger