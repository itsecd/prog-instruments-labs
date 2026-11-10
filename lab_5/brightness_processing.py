"""Модуль для работы с данными и вычислениями."""
import os
import logging
import pandas as pd
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Union
from ImagePathIterator import ImagePathIterator

logger = logging.getLogger(__name__)


def load_image_paths(csv_file: str) -> List[str]:
    """Загрузка путей к изображениям через итератор."""
    if not os.path.exists(csv_file):
        logger.error(f"CSV файл не найден: {csv_file}")
        return []
    
    try:
        iterator = ImagePathIterator(csv_file)

        if len(iterator) == 0:
            logger.warning(f"В файле {csv_file} не найдено валидных путей")
            return []

        image_paths = list(iterator)
        logger.info(f"Успешно загружено {len(image_paths)} путей к изображениям")
        return image_paths

    except Exception as e:
        logger.exception(f"Не удалось загрузить пути из {csv_file}: {e}")
        return []


def brightness_range_calculating(image_path: str) -> Union[float, None]:
    """Вычисление диапазона яркости для одного изображения."""
    if not os.path.exists(image_path):
        logger.warning(f"Изображение не найдено: {image_path}")
        return None

    try:
        with Image.open(image_path) as img:
            img_array = np.array(img)

        if len(img_array.shape) == 2:
            return float(np.max(img_array) - np.min(img_array))
        else:
            channel_ranges = []
            for channel_num in range(img_array.shape[2]):
                channel_data = img_array[:, :, channel_num]
                channel_ranges.append(np.max(channel_data) - np.min(channel_data))
            return float(max(channel_ranges))
             
    except Exception as e:
        logger.error(f"Ошибка обработки изображения {image_path}: {e}")
        return None


def process_brightness(image_paths: List[str], original_csv: str, brightness_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Обработка яркости для всех изображений с сохранением исходных данных."""
    try:
        original_df = pd.read_csv(original_csv, encoding='utf-8')
    except Exception as e:
        logger.critical(f"Не удалось прочитать исходный CSV {original_csv}: {e}")
        return pd.DataFrame()
    
    results = []
    successful = 0

    for image_path in image_paths:
        brightness = brightness_range_calculating(image_path)

        if brightness is not None:
            original_row = original_df[original_df['Absolute path'] == image_path]
            
            if not original_row.empty:
                brightness_range_label = "Не определено"
                if brightness_ranges:
                    for range_label, (min_val, max_val) in brightness_ranges.items():
                        if min_val <= brightness <= max_val:
                            brightness_range_label = range_label
                            break
                
                result_row = {
                    'Absolute_path': image_path,
                    'Relative_path': original_row['Relative path'].iloc[0],
                    'Keyword': original_row['Keyword'].iloc[0],
                    'Brightness_range': brightness,
                    'Brightness_range_label': brightness_range_label
                }
                results.append(result_row)
                successful += 1

    logger.info(f"Обработано {successful} из {len(image_paths)} изображений")
    
    if not results:
        logger.warning("Нет успешно обработанных изображений для формирования DataFrame")
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def sort_by_brightness(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """Сортировка по яркости"""
    logger.debug(f"Сортировка DataFrame по яркости (ascending={ascending})")
    return df.sort_values('Brightness_range', ascending=ascending)


def filter_by_brightness(df: pd.DataFrame, min_range: float = None, max_range: float = None) -> pd.DataFrame:
    """Фильтрация диапазона яркости"""
    logger.debug(f"Фильтрация DataFrame: min_range={min_range}, max_range={max_range}")
    filtered_df = df.copy()

    if min_range is not None:
        filtered_df = filtered_df[
            filtered_df['Brightness_range'] >= min_range
        ]
    if max_range is not None:
        filtered_df = filtered_df[
            filtered_df['Brightness_range'] <= max_range
        ]
        
    return filtered_df


def create_brightness_ranges(ranges_str: str) -> Dict[str, Tuple[int, int]]:
    """Создание словаря диапазонов яркости из строки."""
    ranges_dict = {}
    ranges_list = [r.strip() for r in ranges_str.split(',')]
    
    for range_item in ranges_list:
        if '-' in range_item:
            try:
                min_val, max_val = map(int, range_item.split('-'))
                ranges_dict[range_item] = (min_val, max_val)
            except ValueError:
                logger.warning(f"Некорректный формат диапазона: {range_item}")
                continue
    
    return ranges_dict