import re
import argparse
from typing import List, Dict, Optional, Pattern

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
    """
    Парсит аргументы командной строки.
    
    Returns:
        argparse.Namespace: Объект с разобранными аргументами командной строки
        
    Example:
        >>> args = parse_arguments()
        >>> print(args.filename)
        'data.txt'
    """
    parser = argparse.ArgumentParser(description='Анализ анкет людей')
    parser.add_argument('filename', type=str, help='Имя файла с данными')
    return parser.parse_args()


def read_file(filename) -> Optional[str]:
    """
    Читает содержимое файла в строку.
    
    Args:
        filename (str): Путь к файлу для чтения
        
    Returns:
        Optional[str]: Содержимое файла как строка или None в случае ошибки
        
    Raises:
        FileNotFoundError: Если файл не существует
        IOError: При ошибках ввода-вывода
    """
    try:
        with open(filename, 'r', encoding=DEFAULT_ENCODING) as file:
            return file.read()
    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None


def split_profiles(content) -> List[str]:
    """
    Разделяет содержимое файла на отдельные анкеты.
    
    Args:
        content (str): Полное содержимое файла с анкетами
        
    Returns:
        List[str]: Список строк, каждая из которых представляет одну анкету
        
    Note:
        Разделителем анкет считается шаблон '\\nчисло)\\n'
    """
    profiles = PROFILE_SEPARATOR_PATTERN.split(content)[1:]  # Пропускаем первый элемент (пустой)
    return profiles


def parse_profile(profile_text) -> Dict[str, str]:
    """
    Извлекает данные из текста одной анкеты.
    
    Args:
        profile_text (str): Текст анкеты для анализа
        
    Returns:
        Dict[str, str]: Словарь с извлеченными полями анкеты.
        Возможные ключи: 'surname', 'name', 'gender', 'birth_date', 'contact', 'city'
        
    Example:
        >>> profile = "Фамилия: Иванов\\nИмя: Иван\\n..."
        >>> parse_profile(profile)
        {'surname': 'Иванов', 'name': 'Иван', ...}
    """
    data = {}

    for field, pattern in PROFILE_FIELD_PATTERNS.items():
        match = pattern.search(profile_text)
        if match:
            data[field] = match.group(1).strip()

    return data


def is_21st_century(birth_date) -> bool:
    """
    Проверяет, родился ли человек в 21 веке (после 2000 года).
    
    Args:
        birth_date (str): Строка с датой рождения в различных форматах
        
    Returns:
        bool: True если человек родился после 2000 года, иначе False
        
    Note:
        Поддерживаются форматы дат: дд.мм.гггг, дд/мм/гггг, дд-мм-гггг
        Годы, записанные двумя цифрами, интерпретируются:
        - 00-23 как 2000-2023
        - 24-99 как 1924-1999
    """
    match = DATE_PATTERN.search(birth_date)

    if match:
        day, month, year = match.groups()

        if len(year) == 2:
            prefix = (
                TWENTY_FIRST_CENTURY_PREFIX
                if int(year) <= TWO_DIGIT_YEAR_CUTOFF
                else TWENTIETH_CENTURY_PREFIX
            )
            year = prefix + year

        return int(year) >= TWENTY_FIRST_CENTURY_MIN_YEAR

    return False


def process_profiles(profiles) -> List[str]:
    """
    Обрабатывает все анкеты и находит родившихся в 21 веке.
    
    Args:
        profiles (List[str]): Список текстов анкет
        
    Returns:
        List[str]: Список анкет людей, родившихся после 2000 года
    """
    twenty_first_century_profiles = []

    for profile_text in profiles:
        profile = parse_profile(profile_text)

        if 'birth_date' in profile and is_21st_century(profile['birth_date']):
            twenty_first_century_profiles.append(profile_text)

    return twenty_first_century_profiles


def save_results(profiles, filename=DEFAULT_OUTPUT_FILENAME) -> Optional[str]:
    """
    Сохраняет отобранные анкеты в файл.
    
    Args:
        profiles (List[str]): Список анкет для сохранения
        filename (str): Имя файла для сохранения (по умолчанию '21st_century_profiles.txt')
        
    Returns:
        Optional[str]: Имя файла, в который сохранены данные, или None если нечего сохранять
    """
    if profiles:
        with open(filename, 'w', encoding=DEFAULT_ENCODING) as output_file:
            for i, profile in enumerate(profiles, 1):
                output_file.write(f"{i})\n{profile}\n")
        return filename
    return None


def print_statistics(profiles) -> None:
    """
    Выводит статистику по обработанным анкетам.
    
    Args:
        profiles (List[str]): Список анкет для анализа
    """
    count = len(profiles)
    print(f"Количество людей, родившихся в 21 веке: {count}")


def main() -> None:
    """
    Основная функция программы.
    
    Координирует выполнение всех этапов обработки:
    1. Чтение аргументов командной строки
    2. Чтение файла с данными
    3. Разделение на отдельные анкеты
    4. Обработка и фильтрация анкет
    5. Вывод статистики
    6. Сохранение результатов
    """
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