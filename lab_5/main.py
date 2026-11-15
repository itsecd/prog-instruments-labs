import argparse
import logging
import sys
import image


def setup_logging():
    logger = logging.getLogger("ImageDownloader")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


def get_args() -> list[str]:
    """Parse cmd arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--directory", help="Directory where image will be saved")
    parser.add_argument("-f", "--annotation_file", help="Filename annotation csv file")
    parser.add_argument(
        "-d", "--date_range", help="Date range (format YYYY-MM-DD-YYYY-MM-DD)"
    )
    parser.add_argument("-c", "--count", help="How many images will be download")
    parser.add_argument(
        "-l", "--list", nargs="*", help="Set keywords for image download"
    )
    args = parser.parse_args()

    if (
            args.directory is None
            or args.annotation_file is None
            or args.date_range is None
            or args.count is None
    ):
        logger.warning("Не переданы обязательные аргументы командной строки.")
        return None

    logger.debug(f"Аргументы успешно распарсены: {args}")
    return [
        args.directory,
        args.annotation_file,
        args.date_range,
        args.count,
        args.list,
    ]


def main() -> None:
    """Main function"""
    logger.info("Запуск программы скачивания изображений")
    try:
        directory, annotation_file, date_range, count, keywords = get_args()
        start, end = image.parse_date_range(date_range)
    except TypeError:
        logger.error(
            "Usage: python main.py -o download_dir -f data.csv -d YYYY-MM-DD-YYYY-MM-DD -c count of images (-l [keywords] optional)"
        )
        return
    except Exception as e:
        logger.exception(f"Ошибка при парсинге аргументов: {e}")
        return

    if keywords is None:
        image_paths = image.download_horse_images(start, end, directory, int(count))
    else:
        image_paths = image.download_horse_images(
            start, end, directory, int(count), keywords
        )

    logger.info(f"Скачивание завершено. Всего скачано: {len(image_paths)} изображений.")

    image.create_annotation_csv(image_paths, annotation_file, directory)

    iterator = image.ImageIterator(annotation_file)

    logger.info("Вывод путей к скачанным файлам:")
    for path in iterator:
        logger.debug(path)

    logger.info("Программа успешно завершена")


if __name__ == "__main__":
    main()