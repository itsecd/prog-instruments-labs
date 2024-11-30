import csv
import re

from typing import List, Dict
from json import loads

from path import REGULAR, DATA, V
from checksum import calculate_checksum, serialize_result


def read_data(data_path: str):
    """
    Функция на вход принимает путь по которому лежит csv файл
    Возвращает считанные из него данные
    """
    data = []
    try:
        with open(data_path, "r", encoding="utf-16 le") as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                data.append(row)
            if data:  
                data.pop(0) 
        return data
    except Exception as e:
        print("Ошибка чтения csv файл, ", e)
    

def read_regular(path: str) -> list:
        """
        Функция на вход принимает путь по которому лежит json файл
        Возвращает считанные из него данные
        """
        try:
            with open(path, "r", encoding="UTF-8") as file:
                return loads(file.read())
        except Exception as e:
            print("Ошибка чтения json файл, ", e)


def chek_data(data_path: str, regular_path: str) -> List[int]:
    """
    Функция на вход принимает путь на файлы с датасетом и регулярными выражениями
    Проверяет соотвествие данных из датасета на соответсвие шаблону, заданному регулрными выражениями
    Возвращает строчки не удовлетворяющие требованиям
    """
    data = read_data(data_path)
    regular = read_regular(regular_path)
    invalid = []
    
    for i, row in enumerate(data):
        flag = False
        for val, pattern in zip(row, regular.values()):
            if not re.match(pattern, val):
                flag = True
            else:
                flag = False
            if flag:
                invalid.append(i)
    return invalid
    
if __name__ == "__main__":
    invalid = chek_data(DATA, REGULAR)
    chek_sum = calculate_checksum(invalid)
    serialize_result(V, chek_sum)