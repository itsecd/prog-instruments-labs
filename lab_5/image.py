import csv
import os
import time
import logging
from datetime import date

from icrawler.builtin import GoogleImageCrawler

logger = logging.getLogger("ImageDownloader.image")


class ImageIterator:
    """Class for iteration by images"""

    def __init__(self, source: str):
        """Constructor that accept string path to csv or directory"""
        self.paths = []
        self.counter = 0

        logger.debug(f"Инициализация ImageIterator с источником: {source}")
        if isinstance(source, str):
            if source.endswith(".csv"):
                logger.info(f"Чтение путей из CSV-файла: {source}")
                with open(source, "r", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self.paths.append(row["Absolute Path"])
            else:
                logger.info(f"Поиск изображений в директории: {source}")
                for file in os.listdir(source):
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        self.paths.append(os.path.join(source, file))
            logger.info(f"Найдено путей: {len(self.paths)}")
        else:
            logger.error(f"Неверный тип источника: {type(source)}")
            raise RuntimeError("Cannot create iterable object")

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        """Iteration. Get next element"""
        if self.counter < len(self.paths):
            path = self.paths[self.counter]
            self.counter += 1
            return path
        else:
            raise StopIteration


def parse_date_range(date_range_str: str) -> list[date]:
    """Parse string to list of dates"""
    logger.debug(f"Парсинг диапазона дат: {date_range_str}")
    date_range = list(map(int, date_range_str.split("-")))
    if len(date_range) != 6:
        logger.error(f"Неверный формат даты: {date_range_str}")
        raise ValueError(f"Invalid date format. Usage: YYYY-MM-DD-YYYY-MM-DD")
    return [
        date(date_range[0], date_range[1], date_range[2]),
        date(date_range[3], date_range[4], date_range[5]),
    ]


def download_horse_images(
        start: date,
        end: date,
        directory: str,
        count: int,
        keywords: list[str] = [
            "хобби хорсер",
            "hobby horser",
            "hobbyhorser",
            "horse",
            "pony",
            "horsephoto",
        ],
) -> list[str]:
    """Download images with horses in specified dates range"""
    logger.info(f"Начало скачивания {count} изображений в папку {directory} с {start} по {end}")
    os.makedirs(directory, exist_ok=True)
    downloaded_paths = set()

    if not (os.listdir(directory) is None):
        for file_path in os.listdir(directory):
            if file_path.endswith((".jpg", ".jpeg", ".png")):
                downloaded_paths.add(file_path)

    logger.debug(f"Уже существует файлов в директории: {len(downloaded_paths)}")

    i = 0
    while len(downloaded_paths) < count and i < len(keywords):
        try:
            current_keyword = keywords[i]
            logger.info(f"Попытка скачивания по ключевому слову: '{current_keyword}'")

            filters = dict(type="photo", date=(start, end))
            crawler = GoogleImageCrawler(
                storage={"root_dir": directory}, downloader_threads=4, log_level=50
            )

            crawler.crawl(
                keyword=current_keyword,
                filters=filters,
                max_num=count - len(downloaded_paths),
                file_idx_offset=len(downloaded_paths),
            )

            time.sleep(2)
            if not (os.listdir(directory) is None):
                for file_path in os.listdir(directory):
                    if file_path.endswith((".jpg", ".jpeg", ".png")):
                        downloaded_paths.add(file_path)

            actual_downloaded = len(downloaded_paths)
            logger.info(f"Прогресс: {actual_downloaded}/{count} изображений скачано.")

            i += 1

        except Exception as e:
            logger.error(f"Ошибка при скачивании по слову '{keywords[i]}': {e}", exc_info=True)
            i += 1

    logger.info(f"Скачивание завершено. Итоговое количество: {len(downloaded_paths)}")
    return sorted(downloaded_paths)


def create_annotation_csv(
        image_paths: list[str], annotation_file: str, directory: str
) -> None:
    """Generator csv annotation"""
    logger.info(f"Создание аннотации в файле: {annotation_file}")
    with open(annotation_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Absolute Path", "Relative Path"])

        for path in image_paths:
            abs_path = os.path.abspath(os.path.join(directory, path))
            rel_path = os.path.join(directory, path)
            writer.writerow([abs_path, rel_path])
    logger.info(f"Аннотация успешно создана. Записано {len(image_paths)} строк.")