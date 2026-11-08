import logging
import logging.config


def setup_logging(log_file: str = "app.log") -> None:
    """Настраивает логирование: пишем в файл и в консоль."""
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "my_format": {
                "format": "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "to_console": {
                "class": "logging.StreamHandler",
                "formatter": "my_format",
                "level": "INFO",
            },
            "to_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "my_format",
                "level": "DEBUG",
                "filename": log_file,
                "maxBytes": 5 * 1024 * 1024,
                "backupCount": 3,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["to_console", "to_file"],
        },
        "loggers": {
            "urllib3": {
                "level": "WARNING",
            },
        },
    }
    logging.config.dictConfig(config)