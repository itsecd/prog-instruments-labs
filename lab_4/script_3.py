import os
import shutil
import csv
import random
from multiprocessing import Pool

random_numbers = random.sample(range(0, 10001), 2006)

new_names = [f'{number}.jpg' for number in random_numbers]


def get_old_rel_paths():

    old_path = os.path.relpath('dataset2')

    old_names = os.listdir(old_path)

    old_rel_paths = list(
        map(lambda name: os.path.join(old_path, name), old_names))

    return old_rel_paths


def get_new_rel_paths():

    new_path = os.path.relpath('dataset3')

    new_rel_paths = list(
        map(lambda name: os.path.join(new_path, name), new_names))

    return new_rel_paths


def create_dataset3() -> None:

    if os.path.isdir('dataset3'):
        shutil.rmtree('dataset3')
    os.mkdir('dataset3')
    old_rel_paths = get_old_rel_paths()
    new_rel_paths = get_new_rel_paths()
    zip_paths = zip(old_rel_paths, new_rel_paths)

    with Pool(10) as p:
        p.starmap(shutil.copyfile, zip_paths)


def create_annotation3() -> None:

    old_rel_paths = get_old_rel_paths()

    abs_path = os.path.abspath('dataset3')
    new_full_paths = list(
        map(lambda name: os.path.join(abs_path, name), new_names))

    new_rel_paths = get_new_rel_paths()

    class1 = 'leopard'
    class2 = 'tiger'

    with open('paths3.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\r')
        for full_path, rel_path, old_rel_path in zip(
                new_full_paths, new_rel_paths, old_rel_paths):
            if class1 in old_rel_path:
                class_name = class1
            else:
                class_name = class2
            writer.writerow([full_path, rel_path, class_name])
