import logging
import sys

def setup_logging() -> None:
    """Настраивает глобальное логирование с выводом в консоль и файл."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Вывод в консоль (INFO и выше)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Вывод в файл (DEBUG и выше)
    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Чтобы не дублировать хендлеры при повторном вызове
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)