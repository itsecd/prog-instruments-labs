import argparse
from collections import Counter

def parser() -> argparse.Namespace:
    """
    Парсит аргументы для выполнения задания
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="файл для шифрования",
    )

    parser.add_argument(
        "-k",
        "--key",
        type=str,
        help="ключ",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="output.txt",
        type=str,
        help="Название файла на выходе",
    )


    return parser.parse_args()

def open_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception:
        return ""

def write_file(path : str, text : str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

def cipher_Vigenere(text : str, key : str) -> str:
    """
    Функция для шифрования Виженера
    """
    alphabet = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    len_Alphabet = len(alphabet)

    char_Ind = {char: i for i, char in enumerate(alphabet)}
    index_Char = {i: char for i, char in enumerate(alphabet)}

    key_indexes = [char_Ind[k] for k in key if k in alphabet]

    if not key_indexes:
        return text


    result = []
    key_idx = 0


    for char in text:

        char_upper = char.upper()

        if char_upper in alphabet:

            c_ind = char_Ind[char_upper]
            k_ind = key_indexes[key_idx % len(key_indexes)] 

            new_ind = (c_ind + k_ind) % len_Alphabet
            new_char = index_Char[new_ind]


            result.append(new_char)

            key_idx += 1

        else:
            result.append(char)

    return "".join(result)

def decrypt_Vigenere(text : str, key : str) -> str:

    alphabet = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    len_Alphabet = 33

    char_Ind = {char: i for i, char in enumerate(alphabet)}
    index_Char = {i: char for i, char in enumerate(alphabet)}

    key_indexes = [char_Ind[k] for k in key if k in alphabet]

    if not key_indexes:
        return text


    result = []
    key_idx = 0


    for char in text:

        char_upper = char.upper()

        if char_upper in alphabet:

            c_ind = char_Ind[char_upper]
            k_ind = key_indexes[key_idx % len(key_indexes)] 

            new_ind = (c_ind - k_ind) % len_Alphabet
            new_char = index_Char[new_ind]


            result.append(new_char)

            key_idx += 1

        else:
            result.append(char)

    return "".join(result)


def main() -> None:
    try: 
        args = parser()
        msg = open_file(args.input)
        key = args.key
        write_file(args.output, cipher_Vigenere(msg,key))
        print(f"Зашифрованный текст:\n{cipher_Vigenere(msg,key)}")
        print(f"Расшифрованный текст:\n{decrypt_Vigenere(cipher_Vigenere(msg,key),key)}")


    except Exception as error:
        print(f"Произошла ошибка: {error}")


if __name__ == "__main__":
    main()