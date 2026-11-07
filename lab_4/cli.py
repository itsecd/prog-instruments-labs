import argparse
from crypto import calculate_sha256, find_partial_collision, save_file_hash, verify_file_hash

def run_cli():
    """
    Парсинг командной строки
    :return: ничего
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Путь к исследуемому файлу")
    parser.add_argument("-s", "--save", action="store_true", help="Вычислить и сохранить хеш")
    parser.add_argument("-v", "--verify", help="Путь к файлу .sha256 для верификации")
    parser.add_argument("-c", "--collision", help="Запустить поиск коллизии")

    args = parser.parse_args()

    try:
        if args.collision:
            candidate, col_hash = find_partial_collision(args.collision)
            print(f"\nРезультат: Строка '{candidate}' дает хеш {col_hash}" if candidate else "\nКоллизия не найдена.")
            return

        if not args.file:
            print("Ошибка: Укажите файл через -f или запустите приложение без параметров для GUI")
            return

        if args.save:
            save_path = save_file_hash(args.file)
            print(f"Хеш успешно записан в файл: {save_path}")

        elif args.verify:
            is_intact, current, saved = verify_file_hash(args.file, args.verify)
            print(f"Текущий хеш: {current}\nОжидаемый:    {saved}")
            print("Целостность подтверждена." if is_intact else "Хеши различаются!")
        else:
            print(f"SHA-256 файла: {calculate_sha256(args.file)}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    run_cli()