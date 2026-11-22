import argparse
import random
import sys
from loguru import logger

# Настройка логгера
logger.remove()
logger.add(sys.stdout, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/generator.log", rotation="5 MB",
           retention="7 days", level="DEBUG")


def argument_parsing() -> list[str]:
    """
    Разделение аргументов, введённых пользователем в консоли
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", type=str, help="path to output file")
    args = parser.parse_args()
    logger.debug(
        f"Аргументы командной строки успешно разобраны. Путь: {args.output_file}")
    return [args.output_file]


def generateDRV() -> int:
    """
    Генерация ДСВ (дискретной случайной величины) принимающей значения 0 и 1
    """
    return random.randint(0, 1)


def generateDRVec() -> list[int]:
    """
    Генерация ДСВек (дискретного случайного вектора) из ДСВ выше
    """
    logger.debug("Генерация нового случайного вектора из 128 элементов.")
    res = list()
    for _i in range(128):
        res += [generateDRV()]
    return res


def writeDRVec(output_file: str) -> None:
    """
    Создание файла и запись результата генерации в него
    """
    logger.info(f"Создание и запись случайного вектора в файл: {output_file}")
    try:
        with open(output_file, "+w") as out_file:
            DRVec = generateDRVec()
            DRVec = map(str, DRVec)
            out_file.write("".join(DRVec))
        logger.info(f"Файл {output_file} успешно создан.")
    except PermissionError:
        logger.error(f"Нет прав на запись в файл {output_file}")
        raise


def main():
    output_file = argument_parsing()[0]
    writeDRVec(output_file)


if __name__ == "__main__":
    main()
