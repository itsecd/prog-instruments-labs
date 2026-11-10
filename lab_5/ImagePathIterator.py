import csv
import os
import logging
from typing import Iterator

logger = logging.getLogger(__name__)

class ImagePathIterator:
    """Итератор путей файлов"""
    def __init__(self, annotation_path: str) -> None:
        """
        Args:
            annotation_path: Путь к файлу с аннотацией
        """
        if isinstance(annotation_path, str):
            if annotation_path.endswith('.csv'):
                self.annotation_file = annotation_path
                self.use_annotation = True
            else:
                self.folder_path = annotation_path
                self.use_annotation = False
        else:
            logger.error("Переданный annotation_path не является строкой")
            raise ValueError("Annotation_path must be a string")

        self.current_index = 0
        self.image_paths = []
        self._load_paths()

    def _load_paths(self) -> None:
        """Загружает пути к изображениям из файла с аннотацией или папки"""
        if self.use_annotation:
            try:
                with open(self.annotation_file, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        file_path = row['Absolute path']
                        if os.path.exists(file_path):
                            self.image_paths.append(file_path)
                logger.debug(f"Загружено {len(self.image_paths)} путей из {self.annotation_file}")
            except FileNotFoundError:
                logger.error(f"Файл аннотации не найден: {self.annotation_file}")
                return
            except Exception as e:
                logger.exception(f"Ошибка при чтении файла аннотации {self.annotation_file}: {e}")
        else:
            if os.path.exists(self.folder_path) and os.path.isdir(self.folder_path):
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
                for root, dirs, files in os.walk(self.folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.isfile(file_path):
                            _, file_extension = os.path.splitext(file)
                            file_ext = file_extension.lower()

                            if file_ext in image_extensions:
                                abs_path = os.path.abspath(file_path)
                                self.image_paths.append(abs_path)
                logger.debug(f"Загружено {len(self.image_paths)} путей из папки {self.folder_path}")
            else:
                logger.warning(f"Папка не найдена или не является директорией: {self.folder_path}")

    def __iter__(self) -> Iterator[str]:
        """Возвращает итератор."""
        self.current_index = 0
        return self

    def __next__(self) -> str:
        """Возвращает следующий путь к изображению."""
        if self.current_index < len(self.image_paths):
            path = self.image_paths[self.current_index]
            self.current_index += 1
            return path
        else:
            raise StopIteration

    def __len__(self) -> int:
        """Возвращает количество путей"""
        return len(self.image_paths)