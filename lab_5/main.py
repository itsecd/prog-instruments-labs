import argparse
import logging
from logging.handlers import RotatingFileHandler
from functions import (
    read_annotation_file,
    calculate_aspect_ratios,
    add_range_column,
    sort_by_aspect_ratio,
    filter_by_range,
    create_histogram,
)


def setup_logging():
    logger = logging.getLogger('ImageAnalyzer')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        'app.log', maxBytes=1024 * 1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def main():
    logger = setup_logging()
    logger.info("Запуск приложения для анализа соотношений сторон изображений.")

    try:
        parser = argparse.ArgumentParser(description='Анализ соотношений сторон изображений')
        parser.add_argument('--bins', nargs='+', type=float, required=True,
                            help='Границы диапазонов через пробел')
        parser.add_argument('--filter_range', type=str, required=False,
                            help='Диапазон для фильтрации')
        args = parser.parse_args()

        logger.debug("Аргументы командной строки: bins=%s, filter_range=%s",
                     args.bins, args.filter_range)

        bins = args.bins
        range_filter = args.filter_range

        logger.info("Чтение данных из файла 'annotation.csv'.")
        df = read_annotation_file('annotation.csv')
        logger.info("Успешно загружено %d записей из аннотации.", len(df))

        logger.info("Расчет соотношений сторон.")
        df = calculate_aspect_ratios(df)

        logger.info("Добавление диапазонов.")
        df = add_range_column(df, bins)

        logger.info("Сортировка данных.")
        df_sorted = sort_by_aspect_ratio(df)

        if range_filter:
            logger.info("Фильтрация данных по диапазону: %s", range_filter)
            df_filtered = filter_by_range(df, range_filter)
            logger.info("После фильтрации осталось %d записей.", len(df_filtered))
        else:
            logger.info("Диапазон для фильтрации не указан, шаг пропущен.")

        logger.info("Построение гистограммы.")
        create_histogram(df)
        logger.info("Гистограмма успешно сохранена.")

        df.to_csv('data.csv', index=False)
        logger.info("Обработанные данные сохранены в 'data.csv'.")
        logger.info("Работа программы успешно завершена.")

    except Exception as e:
        logger.error("Критическая ошибка при выполнении программы: %s", e, exc_info=True)


if __name__ == "__main__":
    main()