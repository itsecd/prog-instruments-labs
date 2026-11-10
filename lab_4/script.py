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


def parse_arguments() -> argparse.Namespace:
    """
    Парсинг аргументов командной строки
    """
    parser = argparse.ArgumentParser(description='Поиск дубликатов номеров телефонов в анкетах')
    parser.add_argument('filename', type=str, help='Путь к файлу с данными анкет')
    return parser.parse_args()


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
    Парсинг анкет из текста.
    Использует регулярное выражение для надежного поиска начала новой анкеты.

    Args:
        text: Текст с данными анкет

    Returns:
        Список объектов Profile с данными анкет
    """
    profiles: list[Profile] = []
    current_profile = Profile()
    has_data = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if re.match(r'^\d+\)$', line):
            if has_data:
                profiles.append(current_profile)
            current_profile = Profile()
            has_data = False
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key in FIELD_MAP:
                setattr(current_profile, FIELD_MAP[key], value)
                has_data = True
                if key == "Номер телефона или email":
                    current_profile.phone = extract_phone_number(value)

    if has_data:
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
            phone_dict.setdefault(profile.phone, []).append(profile)

    duplicates = {phone: profs for phone, profs in phone_dict.items() if len(profs) > 1}
    return duplicates


def print_duplicates(duplicates: dict[str, list[Profile]]) -> None:
    """
    Выводит в консоль информацию о найденных дубликатах анкет.

    Args:
        duplicates: Словарь с дубликатами (номер телефона -> список анкет).
    """
    if not duplicates:
        print("Дублирующихся номеров телефонов не найдено.")
        return

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
        print_duplicates(duplicates)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{args.filename}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()