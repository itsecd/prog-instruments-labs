import argparse
import os
import csv
import logging
from icrawler.builtin import BingImageCrawler

logger = logging.getLogger("ImageCrawler")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler('app.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def argparse_keywords() -> argparse.Namespace:
    logger.debug("Начало парсинга аргументов командной строки")
    parser = argparse.ArgumentParser()
    parser.add_argument("file_save", type=str, help="путь к файлу сохранения фото")
    parser.add_argument("annotation", type=str, help="путь к файлу аннотации")
    parser.add_argument("number", type=int, help="количество изображений(от 50 до 100)")
    parser.add_argument("keywords", nargs="*", help="ключевые слова")
    args = parser.parse_args()
    
    if not (50 <= args.number <= 1000):
        logger.error(f"Неверное количество изображений: {args.number}. Допустимо от 50 до 1000.")
        raise ValueError("Количество изображений должно быть от 50 до 1000")
        
    logger.info(f"Аргументы получены. Ключевые слова: {args.keywords}")
    return args


def add_img(keywords: list[str], file_save: str, number: int) -> list[dict[str, str]]:
    data = []
    logger.info(f"Начинаю скачивание. Всего ключевых слов: {len(keywords)}")
    
    for keyword in keywords:
        logger.debug(f"Скачивание по слову: '{keyword}' (по {number} шт.)")
        folder = os.path.join(file_save, keyword)
        crawler = BingImageCrawler(storage={"root_dir": folder})
        
        try:
            crawler.crawl(keyword=keyword, max_num=number)
        except Exception as e:
            logger.error(f"Ошибка при скачивании '{keyword}': {e}")
            continue

        count = 0
        for root, _, files in os.walk(folder):
            for file_1 in files:
                if file_1.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                    abs_path = os.path.abspath(os.path.join(root, file_1))
                    rel_path = os.path.relpath(abs_path, file_save)
                    data.append({
                        "absolute_path": abs_path,
                        "relative_path": rel_path,
                        "keyword": keyword,
                    })
                    count += 1
        logger.info(f"По слову '{keyword}' обработано {count} изображений.")
        
    return data


class Iterator_csv:
    def __init__(self, annotation):
        self.annotation = annotation
        self.data = []
        self.index = 0
        
        try:
            with open(annotation, mode="r", encoding="utf-8", newline="") as f:
                read = csv.DictReader(f)
                self.data = list(read)
            logger.debug(f"Файл {annotation} прочитан. Загружено записей: {len(self.data)}")
        except FileNotFoundError:
            logger.error(f"Файл аннотации {annotation} не найден!")
            raise

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        value = self.data[self.index]
        self.index += 1
        return value


def append_annotation_file(annotation: str, data: list[dict]) -> None:
    logger.info(f"Запись {len(data)} записей в файл: {annotation}")
    fieldnames = ["absolute_path", "relative_path", "keyword"]
    with open(annotation, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        writer.writerows(data)
    logger.info("Запись в файл успешно завершена.")


def main() -> None:
    logger.info("=== Запуск программы ImageCrawler ===")
    
    try:
        args = argparse_keywords()
        data = add_img(args.keywords, args.file_save, args.number)
        append_annotation_file(args.annotation, data)
        
        test = Iterator_csv(args.annotation)
        for val in test:
            print(val)
            
        logger.info("=== Программа успешно завершила работу ===")
        
    except Exception as ex:
        logger.critical(f"Критическая ошибка: {ex}", exc_info=True)


if __name__ == "__main__":
    main()