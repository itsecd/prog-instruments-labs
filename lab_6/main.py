import argparse

from col_timing_graph import *
from luhn import *


def mode_brute_force():
    print("Режим 'brute': Поиск номера карты по хэшу с использованием multiprocessing\n")

    process_count = get_cpu_count()
    print(f"Используем количество процессов: {process_count}")

    found_card = find_card_number()

    if found_card:
        print(f"Карта найдена: {found_card}\n")
        serialization(found_card)
        print("Сохранено в result.json")
    else:
        print("Подходящая карта не найдена.")


def mode_luhn_check():
    print("Режим 'check': Проверка корректности карты по алгоритму Луна\n")

    card = find_card_number()
    print(f"Проверяемый номер: {card}\n")
    print("Результат валидации:\n")
    is_valid(card)


def mode_benchmark():
    print("Режим 'benchmark': Измерение времени подбора карты при различном числе процессов\n")
    print("Запуск бенчмарка...\n")

    proc_list, time_list = collect_times()

    print("Кол-во процессов:")
    print(proc_list)

    print("Время выполнения (сек):")
    print(time_list)
    print("\n")

    get_timing_graph(proc_list, time_list)


def main():
    parser = argparse.ArgumentParser(
        description="Выберите режим запуска:"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("brute", help="Поиск номера карты по хэшу")
    subparsers.add_parser("check", help="Проверка корректности номера карты по алгоритму Луна")
    subparsers.add_parser("benchmark", help="Замер времени подбора при разном числе процессов")

    args = parser.parse_args()

    match args.command:
        case "brute":
            mode_brute_force()
        case "check":
            mode_luhn_check()
        case "benchmark":
            mode_benchmark()


if __name__ == "__main__":
    main()