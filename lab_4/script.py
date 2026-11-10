import argparse
import re
from dataclasses import dataclass

PHONE_PATTERN = re.compile(r'(?:\+7|8)[\s\(\-]*\d{3}[\s\)\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}')

FIELD_MAP = {
    "Фамилия": "surname",
    "Имя": "name",
    "Номер телефона или email": "contact",
}

@dataclass
class Profile:
    """Класс, представляющий анкету пользователя."""
    surname: str = ""
    name: str = ""
    contact: str = ""
    phone: str | None = None


def extract_phone_number(text: str) -> str | None:
    """
    Извлечение номеров телефонов из текста с помощью регулярных выражений

    Args:
        text: Текст для поиска номера телефона

    Returns:
        Нормализованный номер телефона или None, если номер не найден
    """
    phones = PHONE_PATTERN.search(text)
    if phones:
        phone = phones.group()
        cleaned = re.sub(r'[^\d+]', '', phone)
        if cleaned.startswith('8') and len(cleaned) == 11:
            cleaned = '+7' + cleaned[1:]
        return cleaned
    return None


def parse_profiles(text: str) -> list[Profile]:
    """
    Парсинг анкет из текста

    Args:
        text: Текст с данными анкет

    Returns:
        Список объектов Profile с данными анкет
    """
    profiles: list[Profile] = []
    current_profile = Profile()

    for line in text.split('\n'):
        line = line.strip()
        if line.endswith(')'):
            if current_profile.surname:
                profiles.append(current_profile)
            current_profile = Profile()
        elif line.startswith('Фамилия:'):
            current_profile.surname = line.split(':', 1)[1].strip()
        elif line.startswith('Имя:'):
            current_profile.name = line.split(':', 1)[1].strip()
        elif line.startswith('Номер телефона или email:'):
            current_profile.contact = line.split(':', 1)[1].strip()
            current_profile.phone = extract_phone_number(current_profile.contact)

    if current_profile.surname:
        profiles.append(current_profile)
    return profiles


def find_duplicate_phones(profiles: list[Profile]) -> dict[str, list[Profile]]:
    """
    Поиск анкет с дублирующимися номерами телефонов

    Args:
        profiles: Список анкет для анализа

    Returns:
        Словарь где ключи - номера телефонов, значения - списки анкет с этим номером
    """
    phone_dict: dict[str, list[Profile]] = {}

    for profile in profiles:
        if profile.phone:
            if profile.phone in phone_dict:
                phone_dict[profile.phone].append(profile)
            else:
                phone_dict[profile.phone] = [profile]

    duplicates = {phone: profs for phone, profs in phone_dict.items() if len(profs) > 1}
    return duplicates


def main() -> None:
    """
    Основная функция программы
    """
    args = parse_arguments()
    try:
        with open(args.filename, 'r', encoding='utf-8') as file:
            content = file.read()
        profiles = parse_profiles(content)
        duplicates = find_duplicate_phones(profiles)

        if duplicates:
            print("Найдены анкеты с дублирующимися номерами телефонов:")
            print("-" * 40)
            for phone, profiles_list in duplicates.items():
                print(f"\nНомер телефона: {phone}")
                print(f"Количество анкет с этим номером: {len(profiles_list)}")
                print("Анкеты:")
                for i, profile in enumerate(profiles_list, 1):
                    print(f"  {i}. {profile.surname} {profile.name}")
                    print(f"     Контакт: {profile.contact}")
                print("-" * 40)
        else:
            print("Дублирующихся номеров телефонов не найдено.")
    except FileNotFoundError:
        print(f"Ошибка: Файл '{args.filename}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()