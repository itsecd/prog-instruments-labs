import argparse
import sys

from annotate import create_annotation, give_abs_rel_path
from download import download_images
from file_iterator import FileIterator

# Список доступных цветов
list_of_colors = ["red", "green", "blue", "purple"]


def main():

    try:
        parser = argparse.ArgumentParser()

        # Аргументы для путей
        parser.add_argument(
            "-d",
            "--directory_img",
            default="turtle_images",
            help="Путь к папке для сохранения изображений",
        )

        parser.add_argument(
            "-c",
            "--colors",
            nargs="+",
            required=True,
            choices=list_of_colors,
            help="Цвета черепахи",
        )

        parser.add_argument(
            "-a", "--annotation_file", default="annotation.csv", help="Путь к файлу аннотации CSV"
        )

        args = parser.parse_args()

        # Скачивание изображений
        download_images(args.directory_img, args.colors, 50)

        # Создание аннотации
        data_path = give_abs_rel_path(args.directory_img)
        headers = ["absolute_path", "relative_path"]
        create_annotation(args.annotation_file, data_path, headers)

        # Демонстрация работы итератора
        iterator = FileIterator(args.annotation_file)
        for i in iterator:
            print(i)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
