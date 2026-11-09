import argparse
from collections import Counter

def take_key(filenamekey: str) -> dict:
    """Читает файл с ключом и возвращает словарь маппинга символов."""
    key = {}
    with open(filenamekey, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            key_part = parts[0].strip().strip('"').strip("'")
            value_part = parts[1].strip().strip('"').strip("'")
            key[key_part] = value_part
    return key

def read_file_text(filename: str) -> str:
    """Читает файл с текстом."""
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

def write_file_text(filename: str, content: str) -> None:
    """Записывает текстовый контент в файл."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)

def serialize_frequency(frequency: dict) -> str:
    """Преобразует словарь частот в форматированную строку для записи."""
    return "".join(f"{char}:{value}\n" for char, value in frequency.items())

def parse_arguments():
    """Получение аргументов командной строки."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filenametext", type=str, help="Файл с исходным текстом")
    parser.add_argument("filenamekey", type=str, help="Файл с ключом")
    parser.add_argument("filenametextoutput", type=str, help="Файл результата")
    parser.add_argument("filenametextfrequency", type=str, help="Файл для записи частоты")
    return parser.parse_args()

def calculate_frequencies(text: str, key: dict) -> dict:
    """Подсчитывает частоту появления символов из ключа относительно всей длины текста."""
    text_length = len(text)
    if text_length == 0:
        return {}
    
    frequency = {}
    for char in key.keys():
        frequency[char] = 0
        
    for char in text:
        if char in frequency:
            frequency[char] = frequency[char] + 1
            
    for char, count in frequency.items():
        frequency[char] = count / text_length
        
    sorted_frequency = dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
    return sorted_frequency

def decrypt_text(text: str, key: dict) -> str:
    """Выполняет замену символов в тексте согласно переданному словарю-ключу."""
    result = ""
    for char in text:
        if char in key:
            result = result + key[char]
        else:
            result = result + char
    return result

def main():
    args = parse_arguments()
    
    text = read_file_text(args.filenametext)
    key = take_key(args.filenamekey)
    
    frequency = calculate_frequencies(text, key)
    decrypted_text = decrypt_text(text, key)
    
    write_file_text(args.filenametextoutput, decrypted_text)
    write_file_text(args.filenametextfrequency, serialize_frequency(frequency))

if __name__ == "__main__":
    main()
