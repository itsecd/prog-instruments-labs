import pandas as pd
import logging
from PIL import Image
from typing import List
import matplotlib.pyplot as plt

logger = logging.getLogger('ImageAnalyzer.functions')


def read_annotation_file(annotation_file: str) -> pd.DataFrame:
    logger.debug("Попытка чтения файла: %s", annotation_file)
    try:
        df = pd.read_csv(annotation_file)
        return df
    except Exception as e:
        logger.error("Не удалось прочитать файл %s: %s", annotation_file, e)
        raise


def calculate_aspect_ratios(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Начало расчета соотношений сторон.")
    aspect_ratios = []
    error_count = 0

    for path in df['absolute_path']:
        try:
            with Image.open(path) as img:
                width, height = img.size
                aspect_ratios.append(round(width / height, 2))
        except Exception as e:
            logger.warning(
                "Не удалось открыть изображение %s. Использовано значение по умолчанию 1.33. Ошибка: %s",
                path, e,
            )
            aspect_ratios.append(1.33)
            error_count += 1

    if error_count > 0:
        logger.warning("Всего не удалось обработать изображений: %d", error_count)

    df['aspect_ratio'] = aspect_ratios
    return df


def add_range_column(df: pd.DataFrame, bins: List[float]) -> pd.DataFrame:
    try:
        labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
        df['range'] = pd.cut(df['aspect_ratio'], bins=bins, labels=labels)
        return df
    except Exception as e:
        logger.error("Ошибка при добавлении столбца диапазонов: %s", e)
        raise


def sort_by_aspect_ratio(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.sort_values('aspect_ratio')
    except Exception as e:
        logger.error("Ошибка при сортировке: %s", e)
        raise


def filter_by_range(df: pd.DataFrame, range_filter: str) -> pd.DataFrame:
    try:
        return df[df['range'] == range_filter]
    except Exception as e:
        logger.error("Ошибка при фильтрации по диапазону %s: %s", range_filter, e)
        raise


def create_histogram(df: pd.DataFrame, filename: str = 'histogram.png') -> None:
    logger.debug("Создание гистограммы и сохранение в %s", filename)
    plt.figure(figsize=(8, 5))
    df['range'].value_counts().sort_index().plot(kind='bar')
    plt.title('Распределение по соотношению сторон')
    plt.xlabel('Диапазон соотношения')
    plt.ylabel('Количество изображений')
    plt.savefig(filename)
    plt.close()