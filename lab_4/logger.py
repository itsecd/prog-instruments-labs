"""
Простое логирование для приложения перевода.
"""

import logging
import sys
from datetime import datetime


def setup_logging(log_level: str = "INFO"):
    """
    Настройка базового логирования.

    Args:
        log_level: Уровень логирования
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'translation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def get_logger(name: str = "translator"):
    """
    Получить логгер.

    Args:
        name: Имя логгера

    Returns:
        logging.Logger: Объект логгера
    """
    return logging.getLogger(name)