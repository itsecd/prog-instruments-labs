"""Настройка журнала событий приложения."""

import json
import logging
import socket
from datetime import datetime, timezone
from pathlib import Path

APP_NAME = "books_parser"
HOSTNAME = socket.gethostname()


class AuditFormatter(logging.Formatter):
    """Формирует запись с постоянными полями и JSON-данными."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).isoformat(timespec="milliseconds")

        data = dict(getattr(record, "data", {}))
        if record.exc_info:
            data["traceback"] = self.formatException(record.exc_info)

        fields = [
            timestamp,
            record.levelname,
            record.getMessage(),
            APP_NAME,
            HOSTNAME,
            json.dumps(data, ensure_ascii=False, default=str),
        ]
        return "\t".join(fields)


def configure_logging(log_path: Path) -> logging.Logger:
    """Настраивает запись в файл без дублирования обработчиков."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(AuditFormatter())

    logger = logging.getLogger(APP_NAME)

    for old_handler in logger.handlers[:]:
        logger.removeHandler(old_handler)
        old_handler.close()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)

    return logger