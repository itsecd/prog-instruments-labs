import logging
import logging.config
import logging.handlers

LOG_FILE = "lasted.log"


def setup_logging():
    """Настройка логирования: консоль + файл lasted.log."""
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": (
                    "%(asctime)s | %(levelname)-8s | %(name)s | "
                    "%(module)s:%(lineno)d | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": LOG_FILE,
                "maxBytes": 1_000_000,
                "backupCount": 3,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    }
    logging.config.dictConfig(config)
    logging.getLogger(__name__).info("Logging configured. Log file: %s", LOG_FILE)
