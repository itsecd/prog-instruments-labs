import os
from PIL import Image
import threading

import pandas as pd
import numpy as np


def create_DataFrame(data_file_name: str) -> pd.DataFrame:
    """
    Создание dataFrame и переименование колонок
    """
    df = pd.DataFrame(pd.read_csv(data_file_name))
    df.columns = ["Absolute_file_path", "Relative_file_path"]
    return df

def distribution_brightness_columns(df: pd.DataFrame) -> None:
    """
    Добавление колонок со значениями распределения яркости
    """
    files = os.listdir("images")
    blue_average = [0.0]* len(files)
    red_average = [0.0]* len(files)
    green_average = [0.0]* len(files)
    df.insert(2, "Red_average", None)
    df.insert(3, "Green_average", None)
    df.insert(4, "Blue_average", None)
    def process_file(i: int, f: str) -> None:
        path = os.path.join("images", f)
        with Image.open(path) as img:
            img.draft("RGB", (64, 64))
            arr = np.asarray(img.convert("RGB"))
            r, g, b = arr.mean(axis=(0, 1))
        blue_average[i] = b
        red_average[i] = r
        green_average[i] = g

    threads = [threading.Thread(target=process_file, args=(i, f)) for i, f in enumerate(files)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    df["Blue_average"] = blue_average
    df["Green_average"] = green_average
    df["Red_average"] = red_average
