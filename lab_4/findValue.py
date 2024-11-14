import csv
import os.path
from datetime import date
from datetime import timedelta
from loguru import logger
from logger_setup import setup_logger

setup_logger()


def findValueDataset(path: str, _date: date) -> str:
    """возвращает значение по дате из файла dataset"""
    with open(f"{path}/dataset.csv", "r", newline="") as dataset:
        reader = csv.reader(dataset, delimiter=",")
        next(reader)
        for row in reader:
            if row[0] == _date:
                logger.info("Value found.")
                return row[1]
        logger.info("Value not found.")
        return None


def findValueXY(path: str, _date: date) -> str:
    """возвращает значение по дате из файла X"""
    with open(f"{path}/X.csv", "r", newline="") as X:
        readerX = csv.reader(X, delimiter="\n")
        number = 0
        flag = False
        for row in readerX:
            if row[0] == _date:
                flag = True
                break
            number += 1
        if flag == False:
            logger.info("Value not found.")
            return None
        else:
            with open(f"{path}/Y.csv", "r", newline="") as Y:
                readerY = csv.reader(Y, delimiter="\n")
                i = 0
                for row in readerY:
                    if i == number:
                        logger.info("Value found.")
                        return row[0]
                    else:
                        i += 1


def findValueWeek(path: str, _date: date) -> str:
    """возвращает значение по дате из файлов по неделям"""
    year = date.fromisoformat(_date).year
    month = date.fromisoformat(_date).month
    day = date.fromisoformat(_date).day
    day_of_week = date.fromisoformat(_date).isocalendar()[2]
    min_date = date.fromisoformat(_date) - timedelta(days=day_of_week)
    max_date = date.fromisoformat(_date) + timedelta(days=(7 - day_of_week))
    for i in range(0, 8):
        for j in range(0, 8):
            first = min_date + timedelta(days=i)
            last = max_date - timedelta(days=j)
            path_file = f"{path}/{str(first)[0:4]}{str(first)[5:7]}{str(first)[8:10]}_
            {str(last)[0:4]}{str(last)[5:7]}{str(last)[8:10]}.csv"
            if os.path.exists(path_file) == True:
                with open(path_file, "r") as file:
                    reader = csv.reader(file, delimiter=",")
                    for row in reader:
                        if row[0] == _date:
                            logger.info("Value found.")
                            return row[1]
    logger.info("Value not found.")
    return None


def findValueYear(path: str, _date: date) -> str:
    """возвращает значение по дате из файла по годам"""
    year = date.fromisoformat(_date).year
    month = date.fromisoformat(_date).month
    day = date.fromisoformat(_date).day
    day_of_week = date.fromisoformat(_date).isocalendar()[2]
    min_date = date.fromisoformat(f"{year}-01-01")
    max_date = date.fromisoformat(f"{year}-12-31")
    for i in range(0, 365):
        for j in range(0, 365):
            first = min_date + timedelta(days=i)
            last = max_date - timedelta(days=j)
            path_file = f"{path}/{str(first)[0:4]}{str(first)[5:7]}{str(first)[8:10]}_
            {str(last)[0:4]}{str(last)[5:7]}{str(last)[8:10]}.csv"
            if os.path.exists(path_file) == True:
                with open(path_file, "r") as file:
                    reader = csv.reader(file, delimiter=",")
                    for row in reader:
                        if row[0] == _date:
                            logger.info("Value found.")
                            return row[1]
    logger.info("Value not found.")
    return None
 