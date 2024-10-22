import os
import csv
import codecs


def copy_to_new_directory(path_old: str, path_new: str):
    """
    Функция копирования исходного датасета в другую директорию. Принимает на вход старый и новый путь к текстовым
     файлам. Пробегая по каждому, осуществляет копирование.
    """
    for filename in os.listdir(path_old):
        path = os.path.join(path_old, filename)
        if os.path.isfile(path) and filename.endswith('.txt'):
            number = path[-8] + path[-7] + path[-6] + path[-5]
            rev_type = path[-12] + path[-11] + path[-10]
            if rev_type != 'bad':
                rev_type = path[-13] + path[-12] + path[-11] + path[-10]
            file = codecs.open(u'' + path, "r", "utf-8")
            content = file.read()
            file.close()
            file = codecs.open(u'' + path_new + "/" + rev_type + "_" + number + ".txt", "w", "utf-8")
            file.write(content)
            file.close()


def create_new_dir_ann(directory: str):
    """
    Функция создания аннотации для нового датасета. Принимает на вход путь и, пробегая по файлам, заполняет таблицу.
    Метка класса берётся из названия файла.
    """
    columns = ("Path1", "Path2", "Class")
    with open("data1.csv", "w") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(columns)
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if os.path.isfile(path) and path.endswith('.txt'):
                rev_type = path[-12] + path[-11] + path[-10]
                if rev_type == 'bad':
                    file_info = (path,
                                 directory + rev_type + '_' + path[-8] + path[-7] + path[-6] + path[-5], rev_type)
                    writer.writerow(file_info)
                else:
                    rev_type = path[-13] + path[-12] + path[-11] + path[-10]
                    file_info = (path,
                                 directory + rev_type + '_' + path[-8] + path[-7] + path[-6] + path[-5], rev_type)
                    writer.writerow(file_info)
