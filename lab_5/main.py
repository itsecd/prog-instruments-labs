import nist_test
import const
import logging
import logging.config
import sys
import os
from typing import Tuple
from datetime import datetime


def setup_logging():
    os.makedirs("logs", exist_ok=True)

    # обновляем путь к файлу логов динамически
    log_filename = f"logs/test_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # загружаем logging.conf
    logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

    # обновляем fileHandler путь к новому файлу
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.baseFilename = os.path.abspath(log_filename)

    logging.info("Логирование инициализировано")
    logging.debug("Логи будут записываться в файл: %s", log_filename)


def read_file(filename: str) -> str:
    logging.info("Начало чтения файла: %s", filename)

    try:
        with open(filename, "r", encoding="utf-8") as file:
            sequence = file.read()

        logging.info("Файл %s успешно прочитан. Размер: %d символов",
                     filename, len(sequence))
        logging.debug("Первые 50 символов: %.50s...", sequence)

        return sequence

    except FileNotFoundError:
        logging.error("Файл не найден: %s", filename)
        raise
    except IOError as e:
        logging.error("Ошибка ввода-вывода при чтении %s: %s", filename, str(e))
        raise
    except Exception:
        logging.exception("Непредвиденная ошибка при чтении файла")
        raise


def validate_sequence(sequence: str, source: str) -> bool:
    logging.debug("Валидация последовательности из %s", source)

    if not sequence:
        logging.warning("Пустая последовательность из %s", source)
        return False

    valid_chars = {"0", "1"}

    if not all(ch in valid_chars for ch in sequence):
        invalid_chars = set(sequence) - valid_chars
        logging.error("Невалидные символы в %s: %s", source, invalid_chars)
        return False

    logging.info("Последовательность %s валидна. Длина: %d бит",
                 source, len(sequence))

    if len(sequence) < 100:
        logging.warning("Короткая последовательность из %s (%d бит)",
                        source, len(sequence))

    return True


def write_results(freq_cpp: float, freq_java: float,
                  runs_cpp: float, runs_java: float,
                  long_cpp: float, long_java: float,
                  results: str):

    logging.info("Начало записи результатов в файл: %s", results)

    try:
        with open(results, "w", encoding="utf-8") as file:
            file.write("Frequency bitwise test:\n")
            file.write("c++: %s\n" % freq_cpp)
            file.write("java: %s\n\n" % freq_java)

            file.write("A test for identical consecutive bits:\n")
            file.write("c++: %s\n" % runs_cpp)
            file.write("java: %s\n\n" % runs_java)

            file.write("Test for the longest sequence of units in a block:\n")
            file.write("c++: %s\n" % long_cpp)
            file.write("java: %s\n" % long_java)

        logging.info("Результаты успешно записаны в %s", results)

    except IOError as e:
        logging.error("Ошибка записи в файл %s: %s", results, str(e))
        raise
    except Exception:
        logging.exception("Непредвиденная ошибка при записи результатов")
        raise


def perform_tests() -> Tuple[float, float, float, float, float, float]:
    logging.info("Начало выполнения тестов NIST")

    seq_cpp = read_file(const.bin_seq_cpp)
    seq_java = read_file(const.bin_seq_java)

    if not validate_sequence(seq_cpp, "C++"):
        raise ValueError("Невалидная последовательность C++")

    if not validate_sequence(seq_java, "Java"):
        raise ValueError("Невалидная последовательность Java")

    logging.info("Запуск частотного теста")
    freq_cpp = nist_test.frequency_test(seq_cpp)
    freq_java = nist_test.frequency_test(seq_java)

    logging.info("Запуск теста на одинаковые подряд идущие биты")
    runs_cpp = nist_test.runs_test(seq_cpp)
    runs_java = nist_test.runs_test(seq_java)

    logging.info("Запуск теста на самую длинную последовательность единиц")
    long_cpp = nist_test.longest_run_test(seq_cpp)
    long_java = nist_test.longest_run_test(seq_java)

    logging.info("Все тесты NIST завершены")

    return freq_cpp, freq_java, runs_cpp, runs_java, long_cpp, long_java


def main():
    setup_logging()

    logging.info("Запуск программы анализа")

    try:
        test_results = perform_tests()
        write_results(*test_results, const.res)

        logging.info("Анализ завершён успешно. Результаты записаны в %s",
                     const.res)

        freq_cpp, freq_java, *_ = test_results

        if abs(freq_cpp - 0.5) < abs(freq_java - 0.5):
            logging.info("C++ генератор ближе к равномерности")
        else:
            logging.info("Java генератор ближе к равномерности")

    except Exception as e:
        logging.critical("Критическая ошибка: %s", str(e))
        raise

    logging.info("Программа завершена")


if __name__ == "__main__":
    main()
