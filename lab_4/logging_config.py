import logging


def get_info_logger() -> logging.Logger:
    """
    Creataing info logger configuration
    Args:
    - None
    Returns:
    - logging.Logger: Info logger configuration.
    """
    logger = logging.getLogger("info_logger")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - INFO - %(message)s', datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_value_logger() -> logging.Logger:
    """
    Creataing value logger configuration
    Args:
    - None
    Returns:
    - logging.Logger: Value logger configuration.
    """
    logger = logging.getLogger("value_logger")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - VALUE - %(message)s', datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_check_logger() -> logging.Logger:
    """
    Creataing check logger configuration
    Args:
    - None
    Returns:
    - logging.Logger: Check logger configuration.
    """
    logger = logging.getLogger("check_logger")
    if not logger.handlers:
        logger.setLevel(logging.WARNING)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - CHECK - %(message)s', datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

