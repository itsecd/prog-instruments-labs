"""Точка входа для парсера логов."""

import argparse
import logging

from parser import parse_logs
from logger_config import setup_logger


def main() -> None:
    """Главная функция, разбирающая аргументы командной строки."""
    logger = setup_logger(db_path="logs.db")

    arg_parser = argparse.ArgumentParser(
        description="Парсер логов веб-сервера. Извлекает данные с помощью регулярных выражений и собирает статистику.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    arg_parser.add_argument("input_file", help="Путь к исходному файлу с логами (например, server.log)")
    arg_parser.add_argument("-o", "--output", default="parsed_logs.csv", help="Путь для сохранения CSV-файла с валидными строками")
    arg_parser.add_argument("-b", "--broken", default="broken_logs.txt", help="Путь для сохранения файла с битыми строками")
    arg_parser.add_argument("-s", "--stats", default="stats.json", help="Путь для сохранения файла со статистикой (JSON)")
    
    args = arg_parser.parse_args()
    
    logger.info("Запуск парсера с аргументами: %s", args)

    try:
        stats = parse_logs(args.input_file, args.output, args.broken, args.stats)
    except FileNotFoundError:
        logger.error("Файл не найден: %s", args.input_file)
        return
    except PermissionError:
        logger.error("Нет прав на чтение файла: %s", args.input_file)
        return
    except Exception as e:
        logger.exception("Произошла непредвиденная ошибка: %s", e)
        return
    
    logger.info("Всего строк обработано: %d", stats['total_lines_processed'])
    logger.info("Валидных строк (сохранены в %s): %d", args.output, stats['valid_lines'])
    logger.info("Битых строк (сохранены в %s): %d", args.broken, stats['broken_lines'])
    logger.info("Полная статистика сохранена в %s", args.stats)


if __name__ == "__main__":
    main()