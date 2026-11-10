import argparse
import sys
import logging
import matplotlib.pyplot as plt

from logger_config import setup_logging
from brightness_processing import (
    load_image_paths, 
    process_brightness, 
    sort_by_brightness, 
    filter_by_brightness,
    create_brightness_ranges
)
from graphics import (
    create_graph,
    save_dataframe_to_csv, 
    save_plot_to_file
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Анализ диапазона яркости изображений'
    )
    parser.add_argument(
        '--csv', '-c', type=str, required=True, 
        help='Путь к CSV файлу с аннотациями'
    )
    parser.add_argument(
        '--ranges', '-r', type=str, 
        default="0-50,51-100,101-150,151-200,201-255",
        help='Диапазоны яркости в формате "мин-макс,мин-макс,..."'
    )
    return parser.parse_args()


def main() -> None:
    """Основная функция выполнения задания."""
    setup_logging()
    logger.info("=== Запуск программы анализа яркости ===")
    
    try:
        args = parse_arguments()
        logger.debug(f"Аргументы командной строки: CSV={args.csv}, Ranges={args.ranges}")
        
        brightness_ranges = create_brightness_ranges(args.ranges)
        logger.info(f"Создано {len(brightness_ranges)} диапазонов яркости")
        
        image_paths = load_image_paths(args.csv)
        
        if not image_paths:
            logger.warning("Список изображений пуст. Завершение работы.")
            sys.exit(1)
        
        df = process_brightness(image_paths, args.csv, brightness_ranges)
        
        if df.empty:
            logger.warning("DataFrame пуст. Нет данных для анализа.")
            sys.exit(1)
        
        logger.info(f"Статистика: успешно обработано {len(df)} из {len(image_paths)} изображений")
        
        sorted_df = sort_by_brightness(df)
        logger.info("Данные отсортированы по яркости")
        
        filtered_df = filter_by_brightness(df, min_range=100)
        logger.info(f"После фильтрации (min_range=100): {len(filtered_df)} изображений")
        
        logger.info("Распределение по диапазонам яркости:")
        range_stats = df['Brightness_range_label'].value_counts().sort_index()
        
        for range_label, count in range_stats.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {range_label}: {count} изображений ({percentage:.1f}%)")
        
        logger.info("Создание графика...")
        fig = create_graph(df)
        
        logger.info("Сохранение результатов...")
        save_dataframe_to_csv(df, 'brightness_analysis.csv')
        save_plot_to_file(fig, 'brightness_histogram.png')
        
        logger.debug(f"Структура DataFrame: {df.shape[0]} строк, {df.shape[1]} колонок")
        
        if fig is not None:
            logger.info("Отображение графика на экране...")
            plt.show()
        else:
            logger.warning("График не был создан")
            
        logger.info("=== Программа успешно завершила работу ===")
        
    except Exception as e:
        logger.critical(f"Критическая ошибка выполнения: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()