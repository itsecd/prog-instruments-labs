import os
import shutil

import description


def delete_files_and_dir(dir: str) -> None:
    """
    Удаляет все файлы и папки из директории dir
    """
    shutil.rmtree(dir)
    

def make_copy_dataset(path_from: str, path_to: str):
    if not os.path.isdir(path_from):
        raise f"`{path_from}` не является каталогом"
    elif not os.path.isdir(path_to):
        # Если такого пути не существует, он создастся
        path = []
        path.append(path_to[:path_to.rfind("/")]) 
        while not os.path.isdir(path[-1]):
            path.append(path[-1][:path[-1].rfind("/")])
        path.reverse()
        path.pop() # Оставляем в path только несуществующие директории
        for p in path:
            os.mkdir(p) 
    else:
        if path_to[:2] != "./" and path_to[1] != ":":
            path_to = f"./{path_to}"
        delete_files_and_dir(path_to)

    category_name = 0
    if path_from.find("/"):
        category_name = path_from[path_from.rfind("/") + 1:]
    else:
        category_name = path_from

    #  Копирует каталог и все входящие в него файлы и папки в path_to
    shutil.copytree(f"{path_from}", f"{path_to}")

    rename_files(path_to, category_name)
    
    name_file = f"description_{category_name}"
    name_file_copy = f"{name_file}_copy"

    description.make_description(name_file_copy, path_to, "osc")


def rename_files(dir_path: str, category: str) -> None:
    all_items = os.listdir(f"{dir_path}")

    if dir_path[-1] == "/" or dir_path[-1] == "\\":
        dir_path = dir_path[:-1]

    for item in all_items:
        try:
            if os.path.isdir(f"{dir_path}/{item}"):
                rename_files(f"{dir_path}/{item}", item)
                continue

            os.rename(f"{dir_path}/{item}", f"{dir_path}/{category}_{item}")
        except Exception as e:
            print(e)
            continue