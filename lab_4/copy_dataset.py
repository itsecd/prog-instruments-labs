import os
import logging
import shutil

import description

logger = logging.getLogger(__name__)


def delete_files_and_dir(dir: str) -> None:
    """
    Удаляет все файлы и папки из директории dir
    """
    logger.info(f"Attempting to clear directory: {dir}")
    if not os.path.isdir(dir):
        logger.error(f"'{dir}' is not a directory")
        return

    try:
        shutil.rmtree(dir)
        logger.info(f"Successfully cleared directory: {dir}")
    except Exception as e:
        logger.error(f"Failed to clear directory '{dir}': {e}")


def make_copy_dataset(path_from: str, path_to: str):
    """
    Копирует содержимое каталога path_from в path_to, переименовывает файлы,
    и создаёт описание с помощью description.py.
    """
    try:
        logger.info(f"Starting dataset copy from '{path_from}' to '{path_to}'")

        if not os.path.isdir(path_from):
            logger.error(f"`{path_from}` it is not a directory")
            raise f"`{path_from}` it is not a directory"

        elif not os.path.isdir(path_to):
            logger.info(f"Creating directories for '{path_to}'")
            # Если такого пути не существует, он создастся
            path = [path_to[:path_to.rfind("/")]]
            while not os.path.isdir(path[-1]):
                path.append(path[-1][:path[-1].rfind("/")])
            path.reverse()
            path.pop()  # Оставляем в path только несуществующие директории
            for p in path:
                logger.info(f"Create directory {p}")
                os.mkdir(p)
        else:
            if path_to[:2] != "./" and path_to[1] != ":":
                path_to = f"./{path_to}"
            logger.info(f"Target directory '{path_to}' already exists, clearing it")
            delete_files_and_dir(path_to)

        category_name = os.path.basename(path_from.rstrip("/\\"))
        logger.info(f"Category name derived: {category_name}")

        #  Копирует каталог и все входящие в него файлы и папки в path_to
        shutil.copytree(f"{path_from}", f"{path_to}")
        logger.info(f"Copying files from '{path_from}' to '{path_to}'")

        logger.info(f"Renaming files in '{path_to}'")
        rename_files(path_to, category_name)

        name_file = f"description_{category_name}"
        name_file_copy = f"{name_file}_copy"

        logger.info(f"Generating description for '{name_file_copy}' in '{path_to}'")
        description.make_description(name_file_copy, path_to, "osc")
        logger.info("Dataset copy completed successfully")
    except Exception as e:
        logger.error(f"Error during dataset copy: {e}")


def rename_files(dir_path: str, category: str) -> None:
    """
    Переименовывает файлы в директории, добавляя префикс category к имени.
    """
    logger.info(f"Renaming files in directory: {dir_path} with category prefix: {category}")

    all_items = os.listdir(f"{dir_path}")
    dir_path = dir_path.rstrip("/\\")

    for item in all_items:
        try:
            if os.path.isdir(f"{dir_path}/{item}"):
                rename_files(f"{dir_path}/{item}", item)
                continue

            os.rename(f"{dir_path}/{item}", f"{dir_path}/{category}_{item}")
        except Exception as e:
            logger.error(f"Failed to rename '{dir_path}/{item}' in '{dir_path}/{category}_{item}': {e}")
            continue
