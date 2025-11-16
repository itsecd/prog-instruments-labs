import nist_test
import const
import logging
import sys
from typing import Optional, Tuple
import os
from datetime import datetime


def setup_logging():
    os.makedirs('logs', exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_filename = f"logs/test_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info("Логирование инициализировано")
    logging.debug(f"Логи будут записываться в: {log_filename}")


def read_file(filename: str) -> str:
    """
    Читает содержимое файла и возвращает его как строку.

    Параметры:
    filename (str): имя файла для чтения.

    Возвращает:
    str: содержимое файла.
    """
    logging.info(f"Начало чтения файла: {filename}")

    try:
        with open(filename, "r", encoding='utf-8') as file:
            sequence = file.read()
            file_size = len(sequence)

        logging.info(f"Файл {filename} успешно прочитан. Размер: {file_size} символов")
        logging.debug(f"Первые 50 символов: {sequence[:50]}...")

        return sequence

    except FileNotFoundError:
        logging.error(f"Файл не найден: {filename}")
        raise
    except IOError as e:
        logging.error(f"Ошибка ввода-вывода при чтении {filename}: {str(e)}")
        raise
    except Exception as e:
        logging.exception(f"Непредвиденная ошибка при чтении {filename}")
        raise


def write_results(freq_cpp: float, freq_java: float,
                  runs_cpp: float, runs_java: float,
                  long_cpp: float, long_java: float,
                  results: str):
    """
    Записывает результаты тестов в файл.

    Параметры:
    freq_cpp (float): результат частотного теста для C++.
    freq_java (float): результат частотного теста для Java.
    runs_cpp (float): результат теста на одинаковые подряд идущие биты для C++.
    runs_java (float): результат теста на одинаковые подряд идущие биты для Java.
    long_cpp (float): результат теста на самую длинную последовательность единиц в блоке для C++.
    long_java (float): результат теста на самую длинную последовательность единиц в блоке для Java.
    results (str): имя файла для записи результатов.
    """
    logging.info(f"Начало записи результатов в файл: {results}")  # ✅ ДОБАВЛЕНО

    try:
        with open(results, 'w', encoding='utf-8') as file:  # ✅ ИЗМЕНЕНО: добавлена кодировка
            file.write("Frequency bitwise test:\n")
            file.write(f"c++: {freq_cpp}\n")
            file.write(f"java: {freq_java}\n\n")

            file.write("A test for identical consecutive bits:\n")
            file.write(f"c++: {runs_cpp}\n")
            file.write(f"java: {runs_java}\n\n")

            file.write("Test for the longest sequence of units in a block:\n")
            file.write(f"c++: {long_cpp}\n")
            file.write(f"java: {long_java}\n")

        logging.info(f"Результаты успешно записаны в {results}")  # ✅ ДОБАВЛЕНО
        logging.debug(f"Результаты C++: freq={freq_cpp}, runs={runs_cpp}, long={long_cpp}")  # ✅ ДОБАВЛЕНО
        logging.debug(f"Результаты Java: freq={freq_java}, runs={runs_java}, long={long_java}")  # ✅ ДОБАВЛЕНО

    except IOError as e:
        logging.error(f"Ошибка записи в файл {results}: {str(e)}")  # ✅ ДОБАВЛЕНО
        raise
    except Exception as e:
        logging.exception(f"Непредвиденная ошибка при записи результатов")  # ✅ ДОБАВЛЕНО
        raise

def main():
    """
    Основная функция программы.

    Выполняет тесты NIST для последовательностей из C++ и Java,
    записывает результаты в файл.
    """
    freq_cpp = nist_test.frequency_test(read_file(const.bin_seq_cpp))
    freq_java = nist_test.frequency_test(read_file(const.bin_seq_java))
    runs_cpp = nist_test.runs_test(read_file(const.bin_seq_cpp))
    runs_java = nist_test.runs_test(read_file(const.bin_seq_java))
    long_cpp = nist_test.longest_run_test(read_file(const.bin_seq_cpp))
    long_java = nist_test.longest_run_test(read_file(const.bin_seq_java))
    write_results(freq_cpp, freq_java, runs_cpp, runs_java, long_cpp, long_java, const.res)


if __name__ == "__main__":
    main()