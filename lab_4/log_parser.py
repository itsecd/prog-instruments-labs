"""Точка входа для парсера логов."""

import argparse

from parser import parse_logs


def main() -> None:
    """Главная функция, разбирающая аргументы командной строки."""
    arg_parser = argparse.ArgumentParser(
        description="Парсер логов веб-сервера. Извлекает данные с помощью регулярных выражений и собирает статистику.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    arg_parser.add_argument("input_file", help="Путь к исходному файлу с логами (например, server.log)")
    arg_parser.add_argument("-o", "--output", default="parsed_logs.csv", help="Путь для сохранения CSV-файла с валидными строками")
    arg_parser.add_argument("-b", "--broken", default="broken_logs.txt", help="Путь для сохранения файла с битыми строками")
    arg_parser.add_argument("-s", "--stats", default="stats.json", help="Путь для сохранения файла со статистикой (JSON)")
    
    args = arg_parser.parse_args()
    
    print(f"Чтение файла: {args.input_file}...")
    
    try:
        stats = parse_logs(args.input_file, args.output, args.broken, args.stats)
    except FileNotFoundError:
        print(f"Ошибка: файл '{args.input_file}' не найден.")
        return
    except PermissionError:
        print(f"Ошибка: нет прав на чтение файла '{args.input_file}'.")
        return
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        return
    
    print(f"Всего строк обработано: {stats['total_lines_processed']}")
    print(f"Валидных строк (сохранены в {args.output}): {stats['valid_lines']}")
    print(f"Битых строк (сохранены в {args.broken}): {stats['broken_lines']}")
    print(f"Статистика по кодам ответа: {stats['status_codes']}")
    print(f"Полная статистика сохранена в {args.stats}")
    
if __name__ == "__main__":
    main()