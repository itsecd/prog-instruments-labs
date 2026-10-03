import argparse

from hash import avalanche, check_integrity, hash_file, hash_text, load_hash, save_hash


def build_parser() -> argparse.ArgumentParser:
    """Создаёт и настраивает парсер аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Хеш-функции: вычисление SHA-256, проверка целостности и лавинного эффекта"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_parser = subparsers.add_parser("hash", help="Вычислить SHA-256 хеш")
    group = hash_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Путь к файлу")
    group.add_argument("-t", "--text", help="Текст")
    hash_parser.add_argument("-s", "--save", help="Сохранить хеш в файл")

    check_parser = subparsers.add_parser("check", help="Проверить целостность файла")
    check_parser.add_argument("-f", "--file", required=True, help="Путь к файлу")
    check_group = check_parser.add_mutually_exclusive_group(required=True)
    check_group.add_argument("-hf", "--hash-file", help="Файл с сохранённым хешем")
    check_group.add_argument("-hv", "--hash-value", help="Хеш для сравнения")

    av_parser = subparsers.add_parser("avalanche", help="Показать лавинный эффект")
    av_parser.add_argument("text1", help="Первый текст")
    av_parser.add_argument("text2", help="Второй текст")

    return parser


def cmd_hash(args: argparse.Namespace) -> None:
    """
    Обработка команды hash: вычисляет хеш.

    Аргументы: текст либо файл с текстом.
    Возвращает: вычисленный хеш; при указании --save сохраняет его в файл.
    """
    if args.file:
        result = hash_file(args.file)
        if result is None:
            print(f"Ошибка: не удалось прочитать файл '{args.file}'")
            return
        print(f"Хеш файла: {result}")
    else:
        result = hash_text(args.text)
        print(f"Хеш текста: {result}")

    if args.save:
        if save_hash(result, args.save):
            print(f"Хеш сохранён в '{args.save}'")
        else:
            print("Ошибка: не удалось сохранить хеш")


def cmd_check(args: argparse.Namespace) -> None:
    """
    Обработка команды check: проверяет хеш на целостность.

    Аргументы: файл с проверяемым хешем; файл с ожидаемым хешом либо ввод вручную.
    Возвращает: результат проверки (файл изменён / не изменён).
    """
    if args.hash_file:
        expected = load_hash(args.hash_file)
        if expected is None:
            print(f"Ошибка: не удалось прочитать хеш из '{args.hash_file}'")
            return
    else:
        expected = args.hash_value

    if check_integrity(args.file, expected):
        print(f"Файл '{args.file}' не изменён")
    else:
        print(f"Файл '{args.file}' изменён!")


def cmd_avalanche(args: argparse.Namespace) -> None:
    """
    Обработка команды avalanche: показывает лавинный эффект от двух текстов.

    Аргументы: два текста для сравнения.
    Возвращает: хеши обоих текстов, количество и процент различающихся бит.
    """
    h1, h2, diff, percent = avalanche(args.text1, args.text2)
    print(f"\nТекст 1: {args.text1}")
    print(f"Хеш 1: {h1}")
    print(f"\nТекст 2: {args.text2}")
    print(f"Хеш 2: {h2}")
    print(f"\nРазличается бит: {diff} из 256 ({percent:.1f}%)")


def main() -> None:
    """Точка входа: парсит аргументы и вызывает нужную команду."""
    parser = build_parser()
    args = parser.parse_args()

    match args.command:
        case "hash":
            cmd_hash(args)
        case "check":
            cmd_check(args)
        case "avalanche":
            cmd_avalanche(args)


if __name__ == "__main__":
    main()
