import argparse

ALPHABET = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ "


def generate_vigenere_table() -> list[str]:
    """
    Генерирует квадрат Виженера на основе базового алфавита.
    """
    return [ALPHABET[i:] + ALPHABET[:i] for i in range(len(ALPHABET))]


def get_vigenere_char(char: str, key_char: str, table: list[str], mode: str = "encode") -> str:
    """
    Обрабатывает один символ: зашифровывает или расшифровывает.
    """
    char_upper = char.upper()
    
    if char_upper not in ALPHABET:
        return ""

    key_index = ALPHABET.index(key_char.upper())
    
    if mode == "encode":
        char_index = ALPHABET.index(char_upper)
        return table[char_index][key_index]
        
    elif mode == "decode":
        for row in range(len(table)):
            if char_upper == table[row][key_index]:
                return table[row][0]
                
    return ""


def process_text(text: str, key: str, mode: str = "encode") -> str:
    """
    Общая бизнес-логика обхода текста для шифрования и дешифрования.
    """
    table = generate_vigenere_table()
    result_text = []
    key_counter = 0
    key_len = len(key)

    for char in text:
        current_key_char = key[key_counter % key_len]
        
        transformed_char = get_vigenere_char(char, current_key_char, table, mode)
        
        if transformed_char:
            result_text.append(transformed_char)
            key_counter += 1
        else:
            result_text.append(char)
            
    return "".join(result_text)


def read_file(filename: str, strip_whitespace: bool = False) -> str:
    """
    Универсальная функция для чтения файлов.
    """
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
        return content.strip() if strip_whitespace else content


def write_result(text: str, filename: str) -> None:
    """
    Записывает итоговый текст в файл.
    """
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)


def parse_arguments():
    """
    Получение аргументов командной строки.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filenametext", type=str, help="Файл с исходным текстом")
    parser.add_argument("filenamekey", type=str, help="Файл с ключом")
    parser.add_argument("filenametextoutput", type=str, help="Файл для записи результата")
    return parser.parse_args()


def main():
    args = parse_arguments()

    key = read_file(args.filenamekey, strip_whitespace=True)
    text = read_file(args.filenametext)
    
    encoded_text = process_text(text, key, mode="encode")

    # decrypted_text = process_text(encoded_text, key, mode="decode")
    
    write_result(encoded_text, args.filenametextoutput)


if __name__ == "__main__":
    main()
