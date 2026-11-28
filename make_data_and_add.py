import os
import logging

import cv2
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def create_DataFrame(data_file_name: str) -> pd.DataFrame:
    """
    Создание dataFrame и переименование колонок
    """
    logger.info("Creating a DataFrame from file: %s",data_file_name)
    if os.path.exists(data_file_name):
        df = pd.DataFrame(pd.read_csv(data_file_name))
        df.columns = ["Absolute_file_path", "Relative_file_path"]
        return df
    else:
        logger.warning("Failed to create a dataframe.File '%s' does not exist",data_file_name)
        return


def distribution_brightness_columns(df: pd.DataFrame) -> None:
    """
    Добавление колонок со значениями распределения яркости
    """
    logger.info("Adding variation distribution columns to DataFrame (shape: %s)",df.shape)
    blue_average = []
    red_average = []
    green_average = []
    df.insert(2, "Red_average", None)
    df.insert(3, "Green_average", None)
    df.insert(4, "Blue_average", None)
    for f in os.listdir("images"):
        path = os.path.join("images", f)
        if not os.path.exists(path):
            logger.warning("The file from '%s' not found.Continue",path)
            blue_average.append(np.nan)
            red_average.append(np.nan)
            green_average.append(np.nan)
            continue
        img = cv2.imread(path)
        if img is None:
            logger.error("The image from '%s' is not loaded",path)
            blue_average.append(np.nan)
            red_average.append(np.nan)
            green_average.append(np.nan)
            continue
        b, g, r = cv2.split(img)
        blue_average.append(np.mean(b))
        red_average.append(np.mean(r))
        green_average.append(np.mean(g))
    df["Blue_average"] = blue_average
    df["Green_average"] = green_average
    df["Red_average"] = red_average
