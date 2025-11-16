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


def validate_sequence(sequence: str, source: str) -> bool:
    """
    Проверяет валидность бинарной последовательности.

    Параметры:
    sequence (str): последовательность для проверки
    source (str): источник последовательности (для логирования)

    Возвращает:
    bool: True если последовательность валидна
    """
    logging.debug(f"Валидация последовательности из {source}")

    if not sequence:
        logging.warning(f"Пустая последовательность из {source}")
        return False

    valid_chars = set('01')
    if not all(char in valid_chars for char in sequence):
        invalid_chars = set(sequence) - valid_chars
        logging.error(f"Обнаружены невалидные символы в последовательности из {source}: {invalid_chars}")
        return False

    sequence_length = len(sequence)
    logging.info(f"Последовательность из {source} валидна. Длина: {sequence_length} бит")

    if sequence_length < 100:
        logging.warning(f"Короткая последовательность из {source}. Всего {sequence_length} бит")

    return True


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
    logging.info(f"Начало записи результатов в файл: {results}")

    try:
        with open(results, 'w', encoding='utf-8') as file:
            file.write("Frequency bitwise test:\n")
            file.write(f"c++: {freq_cpp}\n")
            file.write(f"java: {freq_java}\n\n")

            file.write("A test for identical consecutive bits:\n")
            file.write(f"c++: {runs_cpp}\n")
            file.write(f"java: {runs_java}\n\n")

            file.write("Test for the longest sequence of units in a block:\n")
            file.write(f"c++: {long_cpp}\n")
            file.write(f"java: {long_java}\n")

        logging.info(f"Результаты успешно записаны в {results}")
        logging.debug(f"Результаты C++: freq={freq_cpp}, runs={runs_cpp}, long={long_cpp}")
        logging.debug(f"Результаты Java: freq={freq_java}, runs={runs_java}, long={long_java}")

    except IOError as e:
        logging.error(f"Ошибка записи в файл {results}: {str(e)}")
        raise
    except Exception as e:
        logging.exception(f"Непредвиденная ошибка при записи результатов")
        raise


def perform_tests() -> Tuple[float, float, float, float, float, float]:
    """
    Выполняет все тесты для обеих последовательностей.

    Возвращает:
    Tuple: результаты всех тестов (freq_cpp, freq_java, runs_cpp, runs_java, long_cpp, long_java)
    """
    logging.info("Начало выполнения тестов NIST")

    # Чтение и валидация последовательностей
    seq_cpp = read_file(const.bin_seq_cpp)
    seq_java = read_file(const.bin_seq_java)

    if not validate_sequence(seq_cpp, "C++"):
        logging.error("Невалидная последовательность C++")
        raise ValueError("Невалидная последовательность C++")

    if not validate_sequence(seq_java, "Java"):
        logging.error("Невалидная последовательность Java")
        raise ValueError("Невалидная последовательность Java")

    # Выполнение тестов
    logging.info("Запуск частотного теста")
    freq_cpp = nist_test.frequency_test(seq_cpp)
    freq_java = nist_test.frequency_test(seq_java)
    logging.debug(f"Частотный тест завершен: C++={freq_cpp}, Java={freq_java}")

    logging.info("Запуск теста на одинаковые подряд идущие биты")
    runs_cpp = nist_test.runs_test(seq_cpp)
    runs_java = nist_test.runs_test(seq_java)
    logging.debug(f"Тест на последовательности завершен: C++={runs_cpp}, Java={runs_java}")

    logging.info("Запуск теста на самую длинную последовательность единиц")
    long_cpp = nist_test.longest_run_test(seq_cpp)
    long_java = nist_test.longest_run_test(seq_java)
    logging.debug(f"Тест на длинные последовательности завершен: C++={long_cpp}, Java={long_java}")

    logging.info("Все тесты NIST успешно выполнены")
    return freq_cpp, freq_java, runs_cpp, runs_java, long_cpp, long_java


def main():
    """
    Основная функция программы.

    Выполняет тесты NIST для последовательностей из C++ и Java,
    записывает результаты в файл.
    """
    setup_logging()

    logging.info("Запуск основной программы анализа последовательностей")

    try:

        test_results = perform_tests()

        write_results(*test_results, const.res)

        freq_cpp, freq_java, runs_cpp, runs_java, long_cpp, long_java = test_results

        logging.info("Анализ завершен успешно")
        logging.info(f"Итоговые результаты записаны в {const.res}")

        if abs(freq_cpp - 0.5) < abs(freq_java - 0.5):
            logging.info("C++ генератор показал лучшую частотную характеристику")
        else:
            logging.info("Java генератор показал лучшую частотную характеристику")

    except Exception as e:
        logging.critical(f"Критическая ошибка во время выполнения программы: {str(e)}")
        raise

    logging.info("Программа завершена")


if __name__ == "__main__":
    main()
