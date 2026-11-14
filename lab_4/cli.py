import argparse
from crypto import calculate_sha256, find_partial_collision, save_file_hash, verify_file_hash


def _create_parser() -> argparse.ArgumentParser:
    """
    Создает и настраивает парсер аргументов командной строки
    :return: настроенный объект ArgumentParser с флагами -f, -s, -v, -c
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Путь к исследуемому файлу")
    parser.add_argument("-s", "--save", action="store_true", help="Вычислить и сохранить хеш")
    parser.add_argument("-v", "--verify", help="Путь к файлу .sha256 для верификации")
    parser.add_argument("-c", "--collision", help="Запустить поиск коллизии")
    return parser


def _handle_collision(prefix: str) -> None:
    """
    Ищет строку, SHA-256 хеш которой начинается с заданного префикса
    :param prefix: искомый префикс хеша
    """

    candidate, col_hash = find_partial_collision(prefix)
    if candidate:
        print(f"\nРезультат: Строка '{candidate}' дает хеш {col_hash}")
    else:
        print("\nКоллизия не найдена.")


def _handle_save(file_path: str) -> None:
    """
    Вычисляет SHA-256 хеш файла и сохраняет его рядом с исходным
    :param file_path: путь к файлу, для которого вычисляется хеш
    """

    save_path = save_file_hash(file_path)
    print(f"Хеш успешно записан в файл: {save_path}")


def _handle_verify(file_path: str, hash_file_path: str) -> None:
    """
    Сравнивает текущий хеш файла с сохраненным ранее и сообщает результат
    :param file_path: путь к проверяемому файлу
    :param hash_file_path: путь к файлу .sha256 с эталонным хешем
    """

    is_intact, current, saved = verify_file_hash(file_path, hash_file_path)
    print(f"Текущий хеш: {current}\nОжидаемый:    {saved}")
    print("Целостность подтверждена." if is_intact else "Хеши различаются!")


def _handle_print_hash(file_path: str) -> None:
    """
    Вычисляет и печатает SHA-256 хеш файла
    :param file_path: путь к исследуемому файлу
    """

    print(f"SHA-256 файла: {calculate_sha256(file_path)}")


def run_cli() -> None:
    """
    Разбирает аргументы командной строки и вызывает обработчик нужной команды
    """

    parser = _create_parser()
    args = parser.parse_args()

    try:
        if args.collision:
            _handle_collision(args.collision)
            return

        if not args.file:
            print("Ошибка: Укажите файл через -f или запустите приложение без параметров для GUI")
            return

        if args.save:
            _handle_save(args.file)
        elif args.verify:
            _handle_verify(args.file, args.verify)
        else:
            _handle_print_hash(args.file)
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    run_cli()