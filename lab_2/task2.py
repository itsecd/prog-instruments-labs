import json


def read_json(file_path: str) -> dict:
    """function, which can get dict from .json file

    Returns:
        dict[str:str]: dictionary with pair (key - value)
    """
    try:
        with open(file_path, 'r', encoding="UTF-8") as file:
            return json.load(file)
    except Exception as e:
       print("Произошла ошибка:", e)


def write_json(file_path: str, content: dict) -> None:
    """function, that writes data to .json file

    Args:
        file_path (str): path to file, that we want to fill
        content (dict): what we want to write in file
    """
    try:
        with open(file_path, 'w', encoding="UTF-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=4)
    except Exception as e:
       print("Произошла ошибка:", e)


def write_txt(file_path: str, data: str) -> None:
    """function, which can write data to .txt file

    Args:
        file_path (str): path to file, which we need to fill
        data (str): what we need to write in file
    """
    with open(file_path, 'w', encoding="UTF-8") as file:
        file.write(data)


def read_txt(file_path: str) -> str:
    """function, which can read data from .txt file

    Args:
        file_path (str): path to file with data

    Returns:
        str: what the file contains
    """
    with open(file_path, "r", encoding="UTF-8") as file:
        return file.read().replace("\n", " \n")


def read_json(file_path: str) -> dict:
    """function, which can get dict from .json file

    Returns:
        dict[str:str]: dictionary with pair (key - value)
    """
    try:
        with open(file_path, 'r', encoding="UTF-8") as file:
            return json.load(file)
    except Exception as e:
       print("Произошла ошибка:", e)



def get_freq(text: str) -> dict:
    """function, which can calculate frequency in text

    Args:
        text (str): text, which we need to process

    Returns:
        dict: dictionary with frequency for current text
    """
    return dict({symbol: text.count(symbol)/len(text) for symbol in set(text)})


def decypher_text(text: str, json_key_content: dict) -> str:
    """decyphers text with given key

    Args:
        text (str): text to decypher
        json_key_content (dict): json that contains alphabet where
        each letter has assigned substitute symbol
        

    Returns:
        str: cyphered text
    """
    result = ""
    key_dict = json_key_content['alphabet']

    for symbol in text:
            if symbol in key_dict:
                result += key_dict[symbol]
            else:
                result += symbol
    return result


def main() -> None:
    text = read_txt("lab_1/task2/cod7.txt")
    sypher_freq_dict = get_freq(text)
    write_json("lab_1/task2/cypher_freq.json", sypher_freq_dict)
    result = decypher_text(text, read_json('lab_1/task2/key.json'))
    write_txt('lab_1/task2/result.txt', result)

        
if __name__ == '__main__':
    main()   
