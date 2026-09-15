"""Модуль фильтрации анкет по корректности дат рождения."""

from __future__ import annotations

import argparse
import re
from datetime import datetime

# Строгий формат даты: ДД.ММ.ГГГГ (только точки)
DATE_PATTERN = re.compile(
    r"^\s*(0?[1-9]|[12][0-9]|3[01])\.(0?[1-9]|1[0-2])\.(19\d{2}|20[0-9]{2})\s*$"
)
BIRTHDATE_PATTERN = re.compile(r"Дата[:\s]*рождения[:\s]*([^\n]+)", re.IGNORECASE)
PROFILE_SPLIT_PATTERN = re.compile(r"\n(?:\d+\)\s*\n)")


def parse_console() -> argparse.Namespace:
    """Парсер для аргументов в консоли."""
    parser = argparse.ArgumentParser(
        description="Фильтрация анкет по корректности дат рождения"
    )
    parser.add_argument("r_file", type=str, help="Путь к исходному файлу с анкетами")
    parser.add_argument(
        "w_file", type=str, help="Путь для сохранения файла с корректными анкетами"
    )
    return parser.parse_args()


def get_current_year() -> int:
    """Возвращает текущий год."""
    return datetime.now().year


def is_valid_real_date(date_str: str) -> bool:
    """Проверяет, является ли дата реально существующей (формат строго ДД.ММ.ГГГГ)."""
    if not DATE_PATTERN.match(date_str):
        return False

    year_match = re.search(r"(\d{4})\s*$", date_str)
    if not year_match:
        return False

    year = int(year_match.group(1))
    current_year = get_current_year()

    if year < 1900 or year > current_year:
        return False

    date_str_clean = date_str.strip()

    try:
        datetime.strptime(date_str_clean, "%d.%m.%Y")
    except ValueError:
        return False
    else:
        return True


def find_incorrect_birthdates(
    text: str,
) -> tuple[list[str], list[str]]:
    """
    Находит анкеты с некорректными датами рождения и разделяет их.

    Возвращает:
        (анкеты_с_некорректными_датами, анкеты_с_корректными_датами)
    """
    profiles = PROFILE_SPLIT_PATTERN.split(text)

    incorrect_profiles: list[str] = []
    correct_profiles: list[str] = []

    for profile in profiles:
        if not profile.strip():
            continue

        date_match = BIRTHDATE_PATTERN.search(profile)

        if date_match:
            date_str = date_match.group(1).strip()
            if is_valid_real_date(date_str):
                correct_profiles.append(profile)
            else:
                incorrect_profiles.append(profile)
        else:
            incorrect_profiles.append(profile)

    return incorrect_profiles, correct_profiles


def read_file(filename: str) -> str:
    """Читает файл и возвращает его содержимое."""
    with open(filename, encoding="utf-8") as file:
        return file.read()


def write_file(filename: str, profiles: list[str]) -> None:
    """Записывает анкеты в файл."""
    with open(filename, "w", encoding="utf-8") as file:
        for i, profile in enumerate(profiles, 1):
            file.write(f"{i})\n{profile}\n\n")


def display_incorrect_profiles(profiles: list[str]) -> None:
    """Выводит анкеты с некорректными датами рождения на экран."""
    if not profiles:
        print("Анкет с некорректными датами рождения не найдено.")
        return

    print(f"\n{'=' * 60}")
    print(f"НАЙДЕНО АНКЕТ С НЕКОРРЕКТНЫМИ ДАТАМИ РОЖДЕНИЯ: {len(profiles)}")
    print(f"{'=' * 60}")

    for i, profile in enumerate(profiles, 1):
        print(f"\n--- Анкета {i} ---")
        print(profile)
        print("-" * 40)


def main() -> None:
    """Точка входа в программу."""
    try:
        args = parse_console()
        text = read_file(args.r_file)

        incorrect_profiles, correct_profiles = find_incorrect_birthdates(text)

        display_incorrect_profiles(incorrect_profiles)
        write_file(args.w_file, correct_profiles)

        print(f"\n{'=' * 60}")
        print("РЕЗУЛЬТАТЫ ОБРАБОТКИ:")
        print(f"{'=' * 60}")
        print(f"Всего обработано анкет: {len(incorrect_profiles) + len(correct_profiles)}")
        print(f"Найдено с некорректными датами: {len(incorrect_profiles)}")
        print(f"Оставлено корректных анкет: {len(correct_profiles)}")
        print(f"Корректные анкеты сохранены в файл: {args.w_file}")

    except FileNotFoundError:
        print("Ошибка: Исходный файл не найден")
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")


if __name__ == "__main__":
    main()