import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Iterator, List
from icrawler.builtin import BingImageCrawler

logger = logging.getLogger("ImageCrawler")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class ImageIterator:
    """Итератор для перебора путей к изображениям"""

    def __init__(self, annotation_file=None, image_dir=None) -> None:
        logger.debug(
            f"Initializing ImageIterator (annotation={annotation_file}, dir={image_dir})"
        )
        if annotation_file:
            self.paths = self._load_from_annotation(annotation_file)
        elif image_dir:
            self.paths = self._load_from_directory(image_dir)
        else:
            logger.error("Neither annotation_file nor image_dir was provided")
            raise ValueError("Необходимо указать либо файл аннотации, либо папку с изображениями")

        self.index = 0
        logger.debug(f"ImageIterator ready, total paths loaded: {len(self.paths)}")

    def _load_from_annotation(self, annotation_file) -> List[str]:
        """Загрузка путей из CSV файла"""
        paths = []
        try:
            with open(annotation_file, "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader, None)  # Пропускаем заголовок
                for row in reader:
                    if row:
                        paths.append(row[0])
        except FileNotFoundError:
            logger.error(f"Annotation file not found: {annotation_file}")
            raise

        if not paths:
            logger.warning(f"Annotation file '{annotation_file}' contains no image records")
        return paths

    def _load_from_directory(self, image_dir) -> List[str]:
        """Загрузка путей из директории"""
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
        paths = []
        for file_path in Path(image_dir).rglob("*"):
            if file_path.suffix.lower() in image_extensions:
                paths.append(str(file_path.absolute()))

        if not paths:
            logger.warning(f"No images found in directory '{image_dir}'")
        return paths

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        if self.index < len(self.paths):
            path = self.paths[self.index]
            self.index += 1
            return path
        logger.debug("Iteration finished")
        raise StopIteration


def download_images(keyword, output_dir, total_images, size_ranges) -> None:
    """Скачивание изображений с фильтрацией по размерам"""
    if total_images <= 0:
        logger.error(f"Invalid total_images value: {total_images}")
        raise ValueError("Количество изображений должно быть больше 0")

    os.makedirs(output_dir, exist_ok=True)
    images_per_range = total_images // len(size_ranges)

    if images_per_range == 0:
        logger.warning(
            f"total_images ({total_images}) is less than ranges count ({len(size_ranges)}), 0 images will be downloaded"
        )

    logger.info(
        f"Starting download: keyword='{keyword}', total={total_images}, ranges={size_ranges}"
    )

    for i, size_range in enumerate(size_ranges):
        range_dir = os.path.join(output_dir, f"range_{i+1}")
        os.makedirs(range_dir, exist_ok=True)

        logger.debug(f"Crawling range [{i+1}/{len(size_ranges)}]: size='{size_range}' into '{range_dir}'")
        crawler = BingImageCrawler(storage={"root_dir": range_dir}, downloader_threads=4)
        filters = {"size": size_range, "type": "photo"}

        try:
            crawler.crawl(
                keyword=keyword,
                filters=filters,
                max_num=images_per_range,
                file_idx_offset=0,
            )
        except Exception as e:
            logger.error(f"Failed to download images for size '{size_range}': {e}")
            raise

        logger.info(f"Completed download for size '{size_range}'")


def create_annotation(output_dir, annotation_file) -> None:
    """Создание CSV аннотации"""
    logger.info(f"Creating annotation file '{annotation_file}' from '{output_dir}'")
    count = 0
    with open(annotation_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Абсолютный путь", "Относительный путь"])

        for root, _, files in os.walk(output_dir):
            for file_name in files:
                if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                    abs_path = os.path.abspath(os.path.join(root, file_name))
                    rel_path = os.path.relpath(abs_path)
                    writer.writerow([abs_path, rel_path])
                    count += 1

    if count == 0:
        logger.warning(f"No images found in '{output_dir}', annotation file is empty")
    else:
        logger.info(f"Annotation saved successfully ({count} records)")


def main() -> None:
    logger.info("=== Starting ImageCrawler application ===")
    parser = argparse.ArgumentParser(description="Скачивание изображений и создание аннотации")
    parser.add_argument("--output", "-o", required=True, help="Путь к папке для сохранения изображений")
    parser.add_argument("--annotation", "-a", required=True, help="Путь к файлу аннотации")
    parser.add_argument("--keyword", "-k", default="rabbit", help="Ключевое слово для поиска")
    parser.add_argument("--total", "-t", type=int, default=100, help="Общее количество изображений")
    parser.add_argument("--ranges", "-r", nargs="+", required=True, help='Диапазоны размеров в формате "large medium small"')

    try:
        args = parser.parse_args()
        logger.debug(f"Parsed CLI arguments: {vars(args)}")

        download_images(args.keyword, args.output, args.total, args.ranges)
        create_annotation(args.output, args.annotation)

        logger.info("Demonstrating ImageIterator (first 5 paths):")
        iterator = ImageIterator(annotation_file=args.annotation)
        for i, path in enumerate(iterator):
            if i >= 5:
                break
            logger.info(f"{i+1}. {path}")

        logger.info("=== Application finished successfully ===")
    except Exception as ex:
        logger.critical(f"Critical application error: {ex}", exc_info=True)


if __name__ == "__main__":
    main()