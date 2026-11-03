import argparse
import re
from pathlib import Path

PHONE_PATTERN = re.compile(r"(?:\+7|8)[\s+]?\(?\d{3}\)?[\s+]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")


def parse_console() -> argparse.Namespace:
    """
    Парсер для аргументов в консоли

    --read_file Путь к файлу для чтения
    --write_file Путь к новому файлу с подходящими анкетами 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("read_file", type=str, help="Чтение файла из консоли")
    parser.add_argument("write_file", type=str, help="Запись нового файла")
    args = parser.parse_args()
    return args


def correct_numbers(text: str) -> list[str]:
    """
    Извлекает анкеты с корректными номерами телефонов
    """
    profiles = re.split(r'\n\s*\d+\)\s*\n', text)
    correct_profiles = []
    for profile in profiles:
        if PHONE_PATTERN.search(profile):
            correct_profiles.append(profile)
    return correct_profiles


def read_file(path: str) -> str:
    """
    Читает содержимое файла
    """
    return Path(path).read_text(encoding="utf-8")


def write_file(path: str, profiles: list[str]) -> None:
    """
    Записывает найденные анкеты в новый файл
    """
    content = "\n\n".join(profiles)
    Path(path).write_text(content, encoding="utf-8")
           



def main() -> None:
    try:
        args = parse_console()
        text = read_file(args.read_file)
        correct_profiles = correct_numbers(text)
        write_file(args.write_file, correct_profiles)
        print(f'Успешно найденных анкет: {len(correct_profiles)}')
        print(f'Записаны в файл: {args.write_file}')

    except FileNotFoundError:
         print("Ошибка. Данный файл не найден")
    except Exception as e:
        print("Ошибка при чтении файла")

    
if __name__=="__main__":
    main()