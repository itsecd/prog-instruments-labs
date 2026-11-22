import logging

import pytest


@pytest.fixture(autouse=True)
def clean_log_parser_logger():
    """
    Чистит хендлеры логгера до и после каждого теста.
    """
    logger = logging.getLogger("log_parser")
    logger.handlers.clear()
    yield
    logger.handlers.clear()