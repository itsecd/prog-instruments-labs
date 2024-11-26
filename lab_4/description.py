import os
import csv


class Description:
    full_path = 0 #полный путь
    path = 0 #относительный путь
    type_class = 0

    def __init__(self, full_path, path, type_class):
        self.full_path = full_path
        self.path = path
        self.type_class = type_class

def about(dir: str) -> list:
    all_elems = os.listdir(dir)
    dirs = []
    description = []
    for elem in all_elems:
        if os.path.isdir(f"{dir}/{elem}"):
            dirs.append(f"{dir}/{elem}")
        else:
            ind = elem.find("_")
            full_path = os.path.abspath(dir+"/"+elem)
            path = os.path.relpath(dir+"/"+elem, os.getcwd())
            type_class = os.path.basename(os.getcwd()+"/"+dir)
            if not (ind == -1):
                type_class = elem[0:ind]
            description.append(Description(full_path, path, type_class))
    for next_dir in dirs:
        description = description + about(next_dir)
    return description



def make_description(name: str, name_dir: str, type_of_file: str) -> None:
    with open(name+".csv", mode="w+", encoding='utf-16') as textFile:
        file_writer = csv.writer(textFile, delimiter = ",")
        file_writer.writerow(["абсолютный путь к файлу", "относительный путь", "метка класса"])
        descriptions_about = about(name_dir)
        if len(type_of_file) != 0:
            all_descriptions = [s for s in descriptions_about if (s.full_path[s.full_path.rfind('.') + 1:].lower() == type_of_file.lower())]
        for description in all_descriptions:
            file_writer.writerow([description.full_path, description.path, description.type_class])

