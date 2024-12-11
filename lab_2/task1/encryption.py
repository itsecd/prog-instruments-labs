import os
import json
import logging
signs = {".", ','}
path = "C:/Users/rodio/PycharmProjects/isb_lab/lab_1/task1/paths.json"
matrix_of_letter = [
    ["а", "б", "в", "г", "д", "е"],
    ["ё", "ж", "з", "и", "й", "к"],
    ["л", "м", "н", "о", "п", "р"],
    ["с", "т", "у", "ф", "х", "ц"],
    ["ч", "ш", "щ", "ъ", "ы", "ь"],
    ["э", "ю", "я", " ", " ", " "],
]

def get_i_j(letter: str) -> tuple:

    try:
        for i in range(0, len(matrix_of_letter)):
            for j in range(0, len(matrix_of_letter[0])):
                if (letter == matrix_of_letter[i][j]):
                    return i, j
    except Exception as e:
        logging.error(f"Ошибка в функции get_i_j(letter): {e}")
        raise


def encryption(message: str) -> str:

    message = message.lower()
    result = ""
    try:
        for letter in message:
            if (letter in signs):
                result += letter
            else:
                place_of_letter = get_i_j(letter)
                if place_of_letter != None:
                    result += str(place_of_letter)
        return result
    except Exception as e:
        logging.error(f"Ошибка в функции encryption(message): {e}")


def message_encryption(file_name: str) -> str:

    try:
        with open(file_name, "r", encoding="utf-8") as file:
            message = file.read()
            encrypted_text = encryption(message)
    except FileNotFoundError:
        print("Файл не найден.")
    else:
        return encrypted_text

def save_message(file_name: str, message) -> None:

    try:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(message)
    except Exception as e:
        logging.error(f"Ошибка в функции get_i_j(letter): {e}")
        raise

def encryption_text() -> None:

    json_data = read_json.read_json_file(path)
    if json_data:
        folder = json_data.get("folder", "")
        path_from = json_data.get("path_from", "")
        path_to = json_data.get("path_to", "")
    if folder and path_from and path_to:
        try:
            encrypted_text = message_encryption(os.path.join(folder, path_from))
            save_message(os.path.join(folder, path_to), encrypted_text)
            print("Текст успешно зашифрован и сохранен в файле.")
        except Exception as e:
            print(f"Произошла ошибка в функции send_encryption_text: {e}")


if __name__ == "__main__":
    encryption_text()