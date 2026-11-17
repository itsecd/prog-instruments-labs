import argparse
import re
from typing import Dict, List, Optional, Pattern

DEFAULT_ENCODING = 'utf-8'
DEFAULT_OUTPUT_FILENAME = '21st_century_profiles.txt'

TWO_DIGIT_YEAR_CUTOFF = 23
TWENTIETH_CENTURY_PREFIX = '19'
TWENTY_FIRST_CENTURY_PREFIX = '20'
TWENTY_FIRST_CENTURY_MIN_YEAR = 2001

PROFILE_SEPARATOR_PATTERN: Pattern[str] = re.compile(r'\n\d+\)\n')
DATE_PATTERN: Pattern[str] = re.compile(r'(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})')

PROFILE_FIELD_PATTERNS: Dict[str, Pattern[str]] = {
    'surname': re.compile(r'Фамилия:\s*([^\n]+)'),
    'name': re.compile(r'Имя:\s*([^\n]+)'),
    'gender': re.compile(r'Пол:\s*([^\n]+)'),
    'birth_date': re.compile(r'Дата рождения:\s*([^\n]+)'),
    'contact': re.compile(r'Номер телефона или email:\s*([^\n]+)'),
    'city': re.compile(r'Город:\s*([^\n]+)'),
}


def parse_arguments() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description='Анализ анкет людей')
    parser.add_argument('filename', type=str, help='Имя файла с данными')
    return parser.parse_args()


def read_file(filename: str, encoding: str = DEFAULT_ENCODING) -> Optional[str]:
    """Читает содержимое текстового файла в строку."""
    try:
        with open(filename, 'r', encoding=encoding) as file:
            return file.read()
    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None


def split_profiles(content: str) -> List[str]:
    """Разделяет содержимое файла на отдельные анкеты по шаблону '\\nчисло)\\n'."""
    return PROFILE_SEPARATOR_PATTERN.split(content)[1:]


def parse_profile(profile_text: str) -> Dict[str, str]:
    """Извлекает структурированные поля из текста одной анкеты."""
    profile_data: Dict[str, str] = {}
    for field, pattern in PROFILE_FIELD_PATTERNS.items():
        match = pattern.search(profile_text)
        if match:
            profile_data[field] = match.group(1).strip()
    return profile_data


def normalize_year(raw_year: str) -> int:
    """Преобразует строковое представление года (2 или 4 цифры) в целое число."""
    if len(raw_year) == 2:
        prefix = (
            TWENTY_FIRST_CENTURY_PREFIX
            if int(raw_year) <= TWO_DIGIT_YEAR_CUTOFF
            else TWENTIETH_CENTURY_PREFIX
        )
        return int(prefix + raw_year)
    return int(raw_year)


def extract_birth_year(birth_date: str) -> Optional[int]:
    """Извлекает год рождения из строки даты или возвращает None."""
    match = DATE_PATTERN.search(birth_date)
    if not match:
        return None
    _, _, raw_year = match.groups()
    return normalize_year(raw_year)


def is_21st_century(birth_date: str) -> bool:
    """Проверяет, относится ли дата рождения к XXI веку (начиная с 2001 года)."""
    year = extract_birth_year(birth_date)
    return year is not None and year >= TWENTY_FIRST_CENTURY_MIN_YEAR


def process_profiles(profiles: List[str]) -> List[str]:
    """Отбирает анкеты людей, родившихся в XXI веке."""
    twenty_first_century_profiles: List[str] = []

    for profile_text in profiles:
        profile = parse_profile(profile_text)
        if 'birth_date' in profile and is_21st_century(profile['birth_date']):
            twenty_first_century_profiles.append(profile_text)

    return twenty_first_century_profiles


def save_results(
    profiles: List[str],
    filename: str = DEFAULT_OUTPUT_FILENAME,
    encoding: str = DEFAULT_ENCODING,
) -> Optional[str]:
    """Сохраняет отобранные анкеты в файл и возвращает его имя."""
    if profiles:
        with open(filename, 'w', encoding=encoding) as output_file:
            for index, profile in enumerate(profiles, start=1):
                output_file.write(f"{index})\n{profile}\n")
        return filename
    return None


def print_statistics(profiles: List[str]) -> None:
    """Выводит в консоль количество отобранных анкет."""
    print(f"Количество людей, родившихся в 21 веке: {len(profiles)}")


def main() -> None:
    """Координирует чтение, фильтрацию и сохранение анкет."""
    args = parse_arguments()

    content = read_file(args.filename)
    if content is None:
        return

    profiles = split_profiles(content)
    twenty_first_century_profiles = process_profiles(profiles)

    print_statistics(twenty_first_century_profiles)

    output_file = save_results(twenty_first_century_profiles)
    if output_file:
        print(f"Анкеты сохранены в файл: {output_file}")


if __name__ == "__main__":
    main()