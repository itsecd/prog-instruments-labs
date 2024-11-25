import os
import csv


def input_data(path_to_the_dataset: str, file_name: str) -> None:
    """
    This function add file paths from folder into the created csv-file

    Args:
        path_to_the_dataset (str): path of default dataset
        file_name (str): name of csv-file
    """
    animals1 = os.listdir(os.path.join(
        path_to_the_dataset, os.listdir(path_to_the_dataset)[0]))
    animals2 = os.listdir(os.path.join(
        path_to_the_dataset, os.listdir(path_to_the_dataset)[1]))
    relative_path1 = os.path.relpath(os.path.join(path_to_the_dataset, os.listdir(
        path_to_the_dataset)[0]), start=path_to_the_dataset)
    relative_path2 = os.path.relpath(os.path.join(path_to_the_dataset, os.listdir(
        path_to_the_dataset)[1]), start=path_to_the_dataset)
    with open(file_name, "a", newline="") as file:
        file_writer = csv.writer(file, delimiter=",", lineterminator='\r')
        for animal in animals1:
            file_writer.writerow(
                [os.path.join(path_to_the_dataset, os.path.join(path_to_the_dataset, os.listdir(path_to_the_dataset)[0]), animal).replace("\\", "/"),
                 os.path.join(relative_path1, animal).replace("\\", "/"),
                 os.path.basename(os.path.normpath(os.path.join(path_to_the_dataset, os.listdir(path_to_the_dataset)[0])))]
            )
        for animal in animals2:
            file_writer.writerow(
                [os.path.join(path_to_the_dataset, os.path.join(path_to_the_dataset, os.listdir(path_to_the_dataset)[1]), animal).replace("\\", "/"),
                 os.path.join(relative_path2, animal).replace("\\", "/"),
                 os.path.basename(os.path.normpath(os.path.join(path_to_the_dataset, os.listdir(path_to_the_dataset)[1])))]
            )


def create_file(file_name: str) -> None:
    """
    This function create a new csv-file (or rewrite existed) with specified by user name

    Args:
        file_name (str): name of creating datasset
    """
    with open(file_name, 'w') as file:
        file_writer = csv.writer(file, delimiter=",", lineterminator="\r")
        file_writer.writerow(["Absolute path", "Relative path", "Type"])
