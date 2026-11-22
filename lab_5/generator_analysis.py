from filereading import read_sequence
from seq_analyze import analyze_sequence
from loguru import logger
import sys

# Настройка логгера
logger.remove()
logger.add(sys.stdout, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="DEBUG")


def main():
    logger.info("Запуск программы анализа генераторов случайных чисел.")

    try:
        logger.debug("Очистка файла statistics.txt")
        file = open("statistics.txt", "w+")
        file.close()

        logger.info("Анализ C++ генератора...")
        cpp_vect = read_sequence("generators/cpp/CppGen.txt")
        analyze_sequence(cpp_vect, "C++")

        logger.info("Анализ Java генератора...")
        java_vect = read_sequence("generators/java/JavaGenerated.txt")
        analyze_sequence(java_vect, "Java")

        logger.info("Анализ Python генератора...")
        python_vect = read_sequence("generators/python/PyGen.txt")
        analyze_sequence(python_vect, "Python")

        logger.info("Программа успешно завершила работу.")
    except Exception as e:
        logger.critical(f"Программа аварийно завершилась: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
