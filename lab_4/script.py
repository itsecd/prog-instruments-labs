import argparse
import re
import sys
from typing import List

EMAIL_FIELD = "Номер телефона или email:"
ALLOWED_DOMAINS = ("gmail.com", "mail.ru", "yandex.ru")
DOMAINS_PATTERN = "|".join(map(re.escape, ALLOWED_DOMAINS))
VALID_EMAIL_PATTERN = re.compile(
    rf"\b[A-Za-z0-9._%+-]{{1,64}}@({DOMAINS_PATTERN})\b"
)
PHONE_HINT_PATTERN = re.compile(r"\+\d|\(\d| \d{3}")
PROFILE_SEPARATOR = re.compile(r"\n\s*\n")
OUTPUT_SEPARATOR = "=" * 80


def read_file(file_path: str) -> str:
    """
    Функция для чтения содержимого файла.

    Args:
        file_path (str): Путь к файлу для чтения

    Returns:
        str: Содержимое файла в виде строки

    Raises:
        FileNotFoundError: Если файл не найден
        PermissionError: Если нет прав для чтения файла
        ValueError: Если файл не в кодировке UTF-8
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Файл '{file_path}' не найден") from e
    except PermissionError as e:
        raise PermissionError(
            f"Нет прав для чтения файла '{file_path}'"
        ) from e
    except UnicodeDecodeError as e:
        raise ValueError(f"Файл '{file_path}' не в кодировке UTF-8") from e


def write_file(file_path: str, data: List[str]) -> None:
    """
    Функция для записи списка анкет в файл.

    Args:
        file_path (str): Путь к входному файлу
        data (List[str]): Список анкет для записи

    Raises:
        PermissionError: Если нет прав для записи в файл
        OSError: Если произошла ошибка ввода-вывода
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(data))
    except PermissionError as e:
        raise PermissionError(
            f"Нет прав для записи в файл '{file_path}'"
        ) from e
    except OSError as e:
        raise OSError(f"Ошибка записи в файл '{file_path}': {e}") from e


def split_into_profiles(content: str) -> List[str]:
    """
    Функция для разбиения содержимого файла на анкеты.

    Args:
        content (str): Содержимое файла с анкетами

    Returns:
        List[str]: Список анкет

    Raises:
        ValueError: Если файл пуст
    """
    if not content.strip():
        raise ValueError("Файл пуст")
    return PROFILE_SEPARATOR.split(content.strip())


def find_email_line(profile: str) -> str | None:
    """
    Функция для поиска строки с телефоном или email в анкете.

    Args:
        profile (str): Текст анкеты

    Returns:
        str | None: Найденная строка или None, если поля нет
    """
    for line in profile.split("\n"):
        if EMAIL_FIELD in line:
            return line
    return None


def has_invalid_email(line: str) -> bool:
    """
    Функция для проверки, содержит ли строка некорректный email.

    Args:
        line (str): Строка с телефоном или email

    Returns:
        bool: True, если в строке некорректный email
    """
    if "@" in line:
        return VALID_EMAIL_PATTERN.search(line) is None
    if any(domain in line for domain in ALLOWED_DOMAINS):
        return PHONE_HINT_PATTERN.search(line) is None
    return False


def extract_profiles_with_invalid_email(content: str) -> List[str]:
    """
    Функция для извлечения анкет с некорректным email.

    Args:
        content (str): Содержимое файла с анкетами

    Returns:
        List[str]: список анкет с некорректными email

    Raises:
        ValueError: Если файл пуст
    """
    profiles = split_into_profiles(content)
    invalid_profiles = []
    for profile in profiles:
        line = find_email_line(profile)
        if line is not None and has_invalid_email(line):
            invalid_profiles.append(profile)
    return invalid_profiles


def remove_profiles_with_invalid_email(
    content: str,
    invalid_profiles: List[str]
) -> List[str]:
    """
    Функция для удаления анкет с некорректным email

    Args:
        content (str): Содержимое файла с анкетами
        invalid_profiles (List[str]): Список анкет для удаления

    Returns:
        List[str]: Список анкет без некорректных email

    Raises:
        ValueError: Если файл пуст
    """
    profiles = split_into_profiles(content)
    invalid_set = set(invalid_profiles)
    return [profile for profile in profiles if profile not in invalid_set]


def main() -> None:
    """
    Основная функция программы.

    Обрабатывает аргументы CLI, находит и удаляет анкеты с некорректыми email
    """
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument(
        'input_file',
        type=str,
        help='Путь к входному файлу с анкетами'
    )

        parser.add_argument(
        'output_file',
        type=str,
        help='Путь к выходному файлу для сохранения результатов'
    )

        args = parser.parse_args()
        content = read_file(args.input_file)
        invalid_profiles = extract_profiles_with_invalid_email(content)

        print("Анкеты с некорректным email:")
        print(OUTPUT_SEPARATOR)
        for profile in invalid_profiles:
            print(profile)
            print(OUTPUT_SEPARATOR)

        print(
            f"\nНайдено {len(invalid_profiles)} анкет с некорректным email.")
        valid_profiles = remove_profiles_with_invalid_email(
            content, invalid_profiles)
        write_file(args.output_file, valid_profiles)
        print(f"\nУдалено {len(invalid_profiles)} анкет с некорректным  email.")
        print(f"Сохранено {len(valid_profiles)} анкет в файл: {args.output_file}")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Ошибка прав доступа: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Ошибка ввода-вывода: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка данных: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
