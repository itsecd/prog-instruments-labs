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
        UnicodeDecodeError: Если ошибка кодировки файла
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл '{file_path}' не найден")
    except PermissionError:
        raise PermissionError(f"Нет прав для чтения файла '{file_path}'")
    except UnicodeDecodeError:
        raise UnicodeDecodeError(
            "Ошибка кодировки файла."
        )


def write_file(file_path: str, data: List[str]) -> None:
    """
    Функция для записи списка анкет в файл.

    Args:
        file_path (str): Путь к входному файлу
        data (List[str]): Список анкет для записи

    Raises:
        PermissionError: Если нет прав для записи в файл
        IOError: Если произошла ошибка ввода-вывода
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(data))
    except PermissionError:
        raise PermissionError(f"Нет прав для записи в файл '{file_path}'")
    except IOError as e:
        raise IOError(f"Ошибка записи в файл '{file_path}': {e}")


def extract_profiles_with_invalid_email(content: str) -> List[str]:
    """
    Функция для извлечения анкет с некорректным email.

    Args:
        content (str): Содержимое файла с анкетами

    Returns:
        List[str]: список анкет с некорректными email

    Raises:
        ValueError: если содержимое файла имеет некорректный формат
    """
    if not content or not content.strip():
        raise ValueError("Файл пуст")

    try:
        profiles = PROFILE_SEPARATOR.split(content.strip())
    except re.error as e:
        raise ValueError(f"Ошибка при чтении анкет: {e}")

    invalid_profiles = []
    for profile in profiles:
        for line in profile.split('\n'):
            if EMAIL_FIELD in line:
                email_phone_line = line
                if '@' in email_phone_line:
                    valid_email_match = VALID_EMAIL_PATTERN.search(
                        email_phone_line)
                    if not valid_email_match:
                        if re.search(r'[А-Яа-я]', email_phone_line):
                            invalid_profiles.append(profile)
                            break
                        elif any(
                                domain in email_phone_line
                                for domain in ALLOWED_DOMAINS):
                            invalid_profiles.append(profile)
                            break
                elif any(
                        domain in email_phone_line
                        for domain in ALLOWED_DOMAINS):
                    if not re.search(r'\+\d|\(\d| \d{3}', email_phone_line):
                        invalid_profiles.append(profile)
                        break
                break
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
        ValueError: Если содержимое файла имеет некорректный формат
    """
    if not content or not content.strip():
        raise ValueError("Файл пуст")

    try:
        profiles = PROFILE_SEPARATOR.split(content.strip())
    except re.error as e:
        raise ValueError(f"Ошибка при чтении анкет: {e}")

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
        for i, profile in enumerate(invalid_profiles, 1):
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
        print(f"Ошибка {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"Ошибка прав доступа {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка данных: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
