"""
Главный файл приложения для перевода PO файлов.
"""

import sys
import argparse
from typing import List

from config import TranslationConfig, BatchConfig
from translation_orchestrator import TranslationOrchestrator
from logger import setup_logging, get_logger


def main():
    """Главная функция приложения."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Настройка логирования
        setup_logging()
        logger = get_logger()

        # Загрузка конфигурации
        config = load_config(args)
        language_codes = load_languages(args)

        # Создание пакетной конфигурации
        batch_config = BatchConfig(language_codes, config)
        batch_config.validate()

        logger.info(f"Начало перевода для языков: {', '.join(language_codes)}")

        # Выполнение перевода
        if config.multi_process:
            run_multi_process(batch_config)
        else:
            run_single_process(batch_config)

        logger.info("Перевод завершен успешно")

    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Создание парсера аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Переводчик PO файлов")

    # Основные аргументы
    parser.add_argument('-l', '--languages', nargs='+', required=True,
                       help='Коды языков для перевода (например: de fr es)')
    parser.add_argument('-d', '--driver-path', required=True,
                       help='Путь к ChromeDriver')
    parser.add_argument('-p', '--locale-path', required=True,
                       help='Путь к папке с локалями')

    # Опциональные аргументы
    parser.add_argument('-i', '--interface-language', default='en',
                       help='Язык интерфейса Google Translate')
    parser.add_argument('-s', '--source-language', default='en',
                       help='Исходный язык для перевода')
    parser.add_argument('-r', '--max-retries', type=int, default=3,
                       help='Максимальное количество повторов')
    parser.add_argument('-m', '--multi-process', action='store_true',
                       help='Использовать многопроцессорность')
    parser.add_argument('--max-processes', type=int, default=10,
                       help='Максимальное количество процессов')
    parser.add_argument('--no-headless', action='store_false', dest='headless',
                       help='Отключить headless режим')

    return parser


def load_config(args: argparse.Namespace) -> TranslationConfig:
    """Загрузка конфигурации из аргументов."""
    return TranslationConfig(
        driver_path=args.driver_path,
        locale_path=args.locale_path,
        headless=args.headless,
        interface_language=args.interface_language,
        source_language=args.source_language,
        max_retries=args.max_retries,
        multi_process=args.multi_process,
        max_processes=args.max_processes
    )


def load_languages(args: argparse.Namespace) -> List[str]:
    """Загрузка списка языков."""
    return args.languages


def run_single_process(batch_config: BatchConfig):
    """Запуск в однопроцессорном режиме."""
    logger = get_logger()

    for language_code in batch_config.language_codes:
        logger.info(f"Перевод языка: {language_code}")

        try:
            with TranslationOrchestrator(batch_config.translation_config) as orchestrator:
                success = orchestrator.translate_language(language_code)

            if success:
                logger.info(f"Успешно переведен: {language_code}")
            else:
                logger.error(f"Ошибка перевода: {language_code}")

        except Exception as e:
            logger.error(f"Ошибка для {language_code}: {e}")


def run_multi_process(batch_config: BatchConfig):
    """Запуск в многопроцессорном режиме."""
    from multiprocessing import Pool
    from itertools import repeat

    logger = get_logger()
    config = batch_config.translation_config

    logger.info(f"Многопроцессорный режим с {config.max_processes} процессами")

    with Pool(processes=min(config.max_processes, len(batch_config.language_codes))) as pool:
        results = pool.starmap(process_language,
                             zip(batch_config.language_codes, repeat(config)))

    success_count = sum(1 for r in results if r)
    logger.info(f"Успешно переведено: {success_count}/{len(results)} языков")


def process_language(language_code: str, config: TranslationConfig) -> bool:
    """Обработка одного языка в пуле процессов."""
    try:
        with TranslationOrchestrator(config) as orchestrator:
            return orchestrator.translate_language(language_code)
    except Exception:
        return False


if __name__ == '__main__':
    main()