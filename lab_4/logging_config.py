import logging


def get_logger(name: str, level: int) -> logging.Logger:
    """
    Creating logger

    Args:
    - name (str): name logger.
    - level (int): level logging (for example, logging.INFO).

    Returns:
    - logging.Logger: logger with given configuration.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_info_logger() -> logging.Logger:
    return get_logger("info_logger", logging.INFO)


def get_error_logger() -> logging.Logger:
    return get_logger("error_logger", logging.ERROR)


def get_debug_logger() -> logging.Logger:
    return get_logger("debug_logger", logging.DEBUG)


def get_warning_logger() -> logging.Logger:
    return get_logger("warning_logger", logging.WARNING)
