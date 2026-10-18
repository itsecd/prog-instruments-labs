import argparse
import csv
import json
import re

import checksum


def read_csv(filename: str) -> dict:
    """Читает CSV-файл и возвращает словарь {имя столбца : список значений}.

    :param filename: путь к CSV-файлу в кодировке UTF-16 с разделителем «;»
    :return: словарь, где ключ — имя столбца, значение — список ячеек этого
        столбца по порядку строк
    raises:
    FileNotFoundError: если файл по указанному пути не найден
    """
    try:
        with open(filename, encoding="utf-16") as f:
            reader = csv.DictReader(f, delimiter=";")
            rows = list(reader)

        result = {}
        for key in reader.fieldnames:
            result[key] = [row[key] for row in rows]
        return result
    except FileNotFoundError as e:
        raise e


def read_json(filename: str) -> dict:
    """Читает JSON с шаблонами и компилирует регулярные выражения.

    param:
    filename: путь к JSON-файлу, где ключ - имя столбца, значение -
        строка регулярного выражения
    return: словарь, где ключ - имя столбца, значение - скомпилированный
        объект регулярного выражения
    raises:
    FileNotFoundError: если файл по указанному пути не найден
    """
    try:
        with open(filename, encoding= 'utf-8') as f:
            return {key : re.compile(value) for key, value in json.load(f).items()}
    except FileNotFoundError as e:
        raise e


def check_reg(csv: dict, patterns: dict) -> set:
    """Возвращает номера строк, где хотя бы одна ячейка не прошла валидацию.

    params:
    csv: словарь {имя столбца : список значений}
    patterns: словарь «имя столбца → скомпилированное регулярное
        выражение», полученный из :func:`read_json`
    return: множество целых номеров строк, содержащих хотя бы одну
        невалидную ячейку
    raises:
    KeyError: если для какого-то столбца из CSV нет шаблона
    """
    result = set()
    try:
        for name, row in csv.items():
            for i in range(len(row)):
                if re.fullmatch(patterns[name], row[i]) is None:
                    result.add(i)
        return result
    except KeyError as e:
        raise e



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_csv", "-r",
                        default="lab_3/3.csv",
                        type = str, help = "csv file to read")
    
    parser.add_argument("--patterns", "-p",
                        default="lab_3/patterns.json",
                        type = str, help = "json file with patterns")
    
    parser.add_argument("--variant", "-v", default=3,
                        type = int, help="number of variant")
    
    parser.add_argument("--write_file", "-w",
                        default="lab_3/result.json",
                        type=str,  help="json file to write result")


    args = parser.parse_args()

    try:
        my_csv = read_csv(args.read_csv)
        patterns = read_json(args.patterns)
        set_lines = list(check_reg(my_csv, patterns))

        my_checksum = checksum.calculate_checksum(set_lines)

        checksum.serialize_result(args.variant, my_checksum, args.write_file)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
