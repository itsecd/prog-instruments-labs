import argparse
import nist_tests
import os
import logging


LOG_FILE = "app.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info(" Запуск программы сравнения последовательностей ")


def parse_arguments():
    """
    Парсинг аргументов
    :return: Объект argparse
    """
    parser = argparse.ArgumentParser(description="Сравнение последовательностей C++ и Java кода")
    parser.add_argument("cpp_file", help="Путь C++ к файлу с сгенерированной последовательностью")
    parser.add_argument("java_file", help="Путь к Java файлу с сгенерированной последовательностью")
    parser.add_argument("results", help="Файл для сохранения результатов тестов")
    return parser.parse_args()


def read_file(filename: str):
    """
    Чтение файла
    :param filename: Путь к файлу
    :return: Последовательность в виде строки
    """
    with open(filename, "r") as file:
        sequence = file.read()
    return sequence


def write_results(freq_cpp, freq_java,
                 runs_cpp, runs_java,
                 long_cpp, long_java,
                 results: str):
    """
    Сохранение результатов тестов в файл
    :param freq_cpp: P-значение частотного побитового теста последовательности из cpp файла
    :param freq_java: P-значение частотного побитового теста последовательности из java файла
    :param runs_cpp: P-значение теста на одинаковые подряд идущие биты последовательности из cpp файла
    :param runs_java: P-значение теста на одинаковые подряд идущие биты последовательности из java файла
    :param long_cpp: P-значение теста на самую длинную послежовательность единиц в блоке последовательности из cpp файла
    :param long_java: P-значение теста на самую длинную послежовательность единиц в блоке последовательности из java файла
    :param results: Путь к файлу для записи результатов
    :return:
    """
    with open(results, 'w') as file:
        file.write("Frequency bitwise test:\n")
        file.write(f"cpp: {freq_cpp}\n")
        file.write(f"java: {freq_java}\n\n")

        file.write("A test for identical consecutive bits:\n")
        file.write(f"cpp: {runs_cpp}\n")
        file.write(f"java: {runs_java}\n\n")

        file.write("Test for the longest sequence of units in a block:\n")
        file.write(f"cpp: {long_cpp}\n")
        file.write(f"java: {long_java}\n")


def main():
    args = parse_arguments()
    logging.debug(f"Аргументы командной строки: cpp_file={args.cpp_file}, java_file={args.java_file}, results={args.results}")

    if not os.path.exists(args.cpp_file):
        logging.error(f"Файл {args.cpp_file} не найден")
        raise FileNotFoundError(f"Файл {args.cpp_file} не найден")

    if not os.path.exists(args.java_file):
        logging.error(f"Файл {args.java_file} не найден")
        raise FileNotFoundError(f"Файл {args.java_file} не найден")

    try:
        seq_cpp = read_file(args.cpp_file)
        seq_java = read_file(args.java_file)

        if len(seq_cpp) != 128 or len(seq_java) != 128:
            raise ValueError("Последовательности должны быть длиной 128 бит!")

        freq_cpp = nist_tests.frequency_monobit_test(seq_cpp)
        freq_java = nist_tests.frequency_monobit_test(seq_java)

        runs_cpp = nist_tests.runs_test(seq_cpp)
        runs_java = nist_tests.runs_test(seq_java)

        long_cpp = nist_tests.longest_run_test(seq_cpp)
        long_java = nist_tests.longest_run_test(seq_java)

        write_results(freq_cpp, freq_java,
                     runs_cpp, runs_java,
                     long_cpp, long_java,
                     args.results)

        print(f"Результаты тестов сохранены в {args.results}")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()