import os
import shutil
import csv


from sources.lab_2_materials.default_dataset_operations import create_file


def make_folder(name: str) -> None:
    """   
    This function create a new folder with specified by user name
    if it is not exist. If the folder exists, function do nothing

    Args:
        name (str): name of creating folder
    """
    if not os.path.isdir(name):
        os.mkdir(name)


def copy_dataset(dataset: str, new_dataset_name: str) -> str:
    """
    This function create a copy of dataset with a specified name

    Args:
        dataset (str): path to default dataset
        new_dataset_name (str): path to the new dataset

    Returns:
        str: path to the new dataset
    """
    make_folder(new_dataset_name)
    animal_types = os.listdir(dataset)
    for animal_type in animal_types:
        animals = os.listdir(os.path.join(dataset, animal_type))
        for animal_photo in animals:
            shutil.copyfile(
                os.path.join(os.path.join(dataset, animal_type), animal_photo),
                os.path.join(new_dataset_name, f"{animal_type}_{animal_photo}")
            )
    return os.path.abspath(new_dataset_name)


def input_data(path_to_the_dataset: str, file_name: str) -> None:
    """
    This function add file paths from folder into the created csv-file

    Args:
        path_to_the_dataset (str): path to the created dataset
        file_name (str): name of csv-file 
    """
    animals = os.listdir(path_to_the_dataset)
    relative_path = os.path.relpath(
        path_to_the_dataset, start=os.path.dirname(path_to_the_dataset))
    with open(file_name, "a", newline="") as file:
        file_writer = csv.writer(file, delimiter=",", lineterminator='\r')
        for animal in animals:
            file_writer.writerow(
                [os.path.join(path_to_the_dataset, animal).replace("\\", "/"),
                 os.path.join(relative_path, animal).replace("\\", "/"),
                 animal.replace(animal[-9:], "")]
            )
