import os
import csv
from typing import List


def get_full_paths(class_name: str) -> List[str]:
    """
    Returns a list of absolute paths for images

    This function returns a list of absolute paths for all images of a certain type
    passed by the function class
    _summary_

    Args:
        class_name (str): _description_

    Returns:
        List[str]: _description_
    """

    full_path = os.path.abspath('dataset')
    class_path = os.path.join(full_path, class_name)
    image_names = os.listdir(class_path)
    image_full_paths = list(
        map(lambda name: os.path.join(class_path, name), image_names))
    return image_full_paths


def get_rel_paths(class_name: str) -> List[str]:
    """
    Returns a list of relative paths to images

    This function returns a list of relative paths relative to the dataset file for
    all images of a certain class passed to the function

    Args:
        class_name (str): _description_

    Returns:
        List[str]: _description_
    """
    rel_path = os.path.relpath('dataset')
    class_path = os.path.join(rel_path, class_name)
    image_names = os.listdir(class_path)
    image_rel_paths = list(
        map(lambda name: os.path.join(class_path, name), image_names))
    return image_rel_paths


def create_annotation() -> None:

    class1 = 'leopard'
    class2 = 'tiger'

    leopard_full_paths = get_full_paths(class1)
    leopard_rel_paths = get_rel_paths(class1)
    tiger_full_paths = get_full_paths(class2)
    tiger_rel_paths = get_rel_paths(class2)

    with open('paths.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\r')
        for full_path, rel_path in zip(leopard_full_paths, leopard_rel_paths):
            writer.writerow([full_path, rel_path, class1])
        for full_path, rel_path in zip(tiger_full_paths, tiger_rel_paths):
            writer.writerow([full_path, rel_path, class2])
