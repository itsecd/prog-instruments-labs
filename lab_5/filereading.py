import sys
from loguru import logger

# Настройка логгера (вывод в консоль и в файл с ротацией)
logger.remove()
logger.add(
    sys.stdout,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time} | {level: <8} | {name}:{function}:{line} - {message}"
)


def read_file(path: str) -> str:
    """
    Чтение строки из файла
    """
    logger.debug(f"Попытка чтения файла: {path}")
    try:
        with open(path, "r+") as file:
            content = file.readline()[:-1]
            logger.info(
                f"Файл {path} успешно прочитан. Длина строки: {len(content)} символов.")
            return content
    except FileNotFoundError:
        logger.error(f"Файл не найден по пути: {path}")
        raise
    except Exception as e:
        logger.critical(f"Критическая ошибка при чтении файла {path}: {e}")
        raise


def read_vector(line: str) -> list[bool]:
    """
    Чтение вектора из строки
    """
    result = []
    for symb in line:
        if symb == "0":
            result += [False]
        elif symb == "1":
            result += [True]
        else:
            logger.error(
                f"Некорректный формат файла. Встречен неизвестный символ: '{symb}'")
            raise ValueError(
                "Uncorrect file format! Symbol: " + symb + "is used!")

    logger.debug(
        f"Вектор успешно распарсен. Количество элементов: {len(result)}")
    return result


def read_sequence(path: str) -> list[bool]:
    """
    Чтение вектора из файла
    """
    return read_vector(read_file(path))
