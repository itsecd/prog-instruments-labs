import os
import random
import csv
import shutil
import logging

from PyQt6 import QtWidgets
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QWidget


logging.basicConfig(level = logging.DEBUG, 
                    format = "%(asctime)s - %(levelname)s - %(message)s",
                    handlers = [
                        logging.FileHandler("logging.log"),
                        logging.StreamHandler()
                    ])


def create_dataset(main_window: QWidget) -> None:
    select_folder = main_window.select_folder
    folderpath = QtWidgets.QFileDialog.getExistingDirectory(
        main_window, "Выберите папку")
    logging.info(f"Вы выбрали: {folderpath}")
    main_window.next_folder = folderpath

    new_folder_path = main_window.next_folder

    annotation_file = "annotation.csv"

    logging.info("Создание файла аннотации данного датасета")

    if os.path.exists(select_folder):
        logging.error("Исходная папка не найдена")
        return None

    logging.info(f"Создание аннотации в папке: {new_folder_path}/{annotation_file}")

    with open(new_folder_path + "/" + annotation_file, "w", newline="",
               encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)

        for folder in os.listdir(select_folder):
            if folder != "tiger" and folder != "leopard":
                logging.warning(f"Не правильная папка: {select_folder}")
                continue

            path_folder = os.path.join(select_folder, folder)

            for img in os.listdir(path_folder):
                img_path = os.path.join(path_folder, img)

                absolute_path = os.path.abspath(img_path)

                relative_path = os.path.relpath(img_path, select_folder)

                csv_writer.writerow([absolute_path, relative_path, folder])
                logging.info(f"Добавлено изображение: {relative_path}")
    logging.info("Файл аннотации создан")


def on_clicked_button(main_windows: QWidget) -> str:
    folderpath = QtWidgets.QFileDialog.getExistingDirectory(
        main_windows, "Выберите папку")
    logging.info(f"Вы выбрали: {folderpath}")
    main_windows.select_folder = folderpath


def on_clicked_button_for_dataset(main_window: QWidget) -> None:
    logging.info("Начало создание датасета")
    create_dataset(main_window.select_folder, main_window.next_folder)


def copy_dataset_with_random(main_window: QWidget) -> None:
    source_dataset = main_window.select_folder
    folderpath = QtWidgets.QFileDialog.getExistingDirectory(
        main_window, "Выберите папку")
    logging.info(f"Вы выбрали: {folderpath}")
    main_window.next_folder = folderpath

    target_dataset = main_window.next_folder

    os.makedirs(target_dataset, exist_ok=True)
    annotation_file = os.path.join(target_dataset, 'annotation.csv')

    with open(annotation_file, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)

        for folder in os.listdir(source_dataset):
            path_folder = os.path.join(source_dataset, folder)

            logging.info(f"Копирование файлов из папки: {folder}")

            for img in os.listdir(path_folder):
                img_path = os.path.join(path_folder, img)

                new_img_name = f"{random.randint(1, 10000)}.jpg"

                target_img_path = os.path.join(target_dataset, new_img_name)

                shutil.copy(img_path, target_img_path)

                absolute_path = os.path.abspath(target_img_path)

                relative_path = os.path.relpath(
                    target_img_path, target_dataset)

                csv_writer.writerow([absolute_path, relative_path, folder])
                logging.info(f"Скопировано изображение: {relative_path}")

    logging.info("Датасет скопировался")


class Iterator:

    def __init__(self, class_label, dataset_path) -> None:
        self.class_label = class_label
        self.dataset_path = dataset_path
        self.class_path = os.path.join(self.dataset_path, class_label)
        self.instances = self.get_instances()

    def get_instances(self) -> list:
        if not os.path.exists(self.class_path):
            logging.error(f"Папка {self.class_label} не найдена.")
            return None

        instances = os.listdir(self.class_path)
        random.shuffle(instances)
        logging.debug(f"Получены экземпляры для {self.class_label}: {instances}")
        return instances

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if not self.instances:
            logging.debug("Экземпляры закончились.")
            raise StopIteration("Экземпляры закончились.")
        return os.path.join(self.class_path, self.instances.pop(0))


def next_animal(main_window: QWidget, class_label: str) -> None:
    manager = Iterator(class_label, main_window.select_folder)
    logging.debug(f"Попытка получить следующее изображение для: {class_label}.")
    try:
        image_path = manager.__next__()
        logging.debug(f"Получен путь к изображению: {image_path}.")
        main_window.current_image = QPixmap(image_path)
        main_window.label.setPixmap(main_window.current_image.scaled(400, 400))
        logging.info(f"Показано следующее изображение: {class_label}.")
    except StopIteration:
        logging.warning(f"Нет доступных изображений для: {class_label}.")
