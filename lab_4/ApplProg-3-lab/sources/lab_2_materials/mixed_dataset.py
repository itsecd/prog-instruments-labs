import os
import shutil
import csv
from random import sample


from sources.lab_2_materials.united_dataset import create_file, make_folder


def copy_and_rename_dataset(default_dataset: str, new_dataset_name: str, csv_file_name: str) -> str:
    """    
    This function create a copy of specified dataset

    Args:
        default_dataset (str): path to the default dataset
        new_dataset_name (str): path to the new dataset with pictures in random prder
        csv_file_name (str): name of csv-file which will contain the data of the new dataset

    Returns:
        str: path to tne new dataset
    """
    random_number = sample(list(range(10000)), 2500)
    make_folder(new_dataset_name)
    animal_types = os.listdir(default_dataset)
    relative_path = os.path.relpath(new_dataset_name)
    counter = 0
    for animal_type in animal_types:
        animals = os.listdir(os.path.join(default_dataset, animal_type))
        for animal_photo in animals:
            random_name = str(random_number[counter]).zfill(5)
            shutil.copyfile(os.path.join(os.path.join(default_dataset, animal_type), animal_photo),
                            os.path.join(new_dataset_name,
                                         random_name + ".jpg")
                            )
            counter += 1
        with open(f"{csv_file_name}.csv", 'a', newline='') as file:
            file_writer = csv.writer(file, delimiter=",", lineterminator='\r')
            for animal in animals:
                file_writer.writerow(
                    [os.path.join(new_dataset_name, animal).replace("\\", "/"),
                     os.path.join(os.path.normpath(relative_path), animal).replace("\\", "/"),
                     animal_type]
                )

    return os.path.abspath(new_dataset_name)