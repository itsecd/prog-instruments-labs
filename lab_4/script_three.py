import csv
import os
import random

from tqdm import tqdm 


def script_three(path_dir: str) -> str:
    os.chdir(path_dir)
    file_name = "test_csv_three.csv"
    names = [i for i in range(10000)]
    with open(file_name, mode="w") as w_file:
        writer = csv.writer(w_file, dialect='excel', delimiter=",", lineterminator="\r")
        writer.writerow(("absolut path", "relativ path", "quote")) 
        pbar = tqdm(os.listdir("dataset_two"), ncols=100, colour='green')
        for element in pbar:
            name = random.choice(names)
            names.remove(name)   
            with open(os.path.join("dataset_two", element), "rb") as f:
                text = f.read()
            with open(os.path.join("dataset_three", str(name) + ".txt"), "wb") as f:
                f.write(text)
            directory = os.path.join(path_dir, "dataset_three", str(name) + ".txt")
            writer.writerow([directory, os.path.join("dataset_two", element), element[0]])