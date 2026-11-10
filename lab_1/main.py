import argparse
import calendar
import re

# Кокшина Анна Александровна 6211-100503


def correct_data(data: str) -> bool:
    """Функция проверяет корректность написания даты"""
    if not re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{4}", data):
        return False
    day, month, year = map(int, re.split(r"[/.-]+", data))
    if not (1 <= month <= 12):
        return False
    return 1 <= day <= calendar.monthrange(year, month)[1] and 1900 <= year <= 2025


def correct_phone(phone: str) -> bool:
    """Функция проверяет корректность номера телефона"""
    return bool(
        re.fullmatch(r"(?:8|\+7)\s?\(?\d{3}\)?\s?\d{3}[\s-]?\d{2}[\s-]?\d{2}", phone)
    )


def correct_email(email: str) -> bool:
    """Функция проверяет корректность указанной почты"""
    return bool(
        re.fullmatch(r"[A-Za-z0-9._%+-]{1,64}@(gmail\.com|mail\.ru|yandex\.ru)", email)
    )


def correct_person(person: list[str]) -> bool:
    """Функция проверяет анкету человека"""
    if person[0][0].islower() or person[1][0].islower():
        return False
    if person[2] not in [
        "М",
        "м",
        "Мужской",
        "мужской",
        "Ж",
        "ж",
        "Женский",
        "женский",
    ]:
        return False
    if not correct_data(person[3]):
        return False
    return bool(correct_phone(person[4]) or correct_email(person[4]))


def read_file(file_name: str) -> list[str]:
    """Считывает файл и возвращает строки, если файл не открыт - None"""
    try:
        with open(file_name, encoding="utf8") as file:
            return file.readlines()
    except FileNotFoundError as err:
        raise FileNotFoundError from err


def print_result(names: dict[str, int]) -> None:
    """Функция находит часто встречающееся имя и выводит его"""
    max_count = 0
    max_name = ""
    for name, count in names.items():
        if max_count < count:
            max_count = count
            max_name = name
    print(f"Самое частое имя {max_name} и количество повторений: {max_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help="Укажите путь к файлу")
    args = parser.parse_args()
    try:
        data = read_file(args.file_name)
    except FileNotFoundError:
        print("Ошибка открытия файла")
        exit(1)
    person = []
    names = {}
    for line in data:
        if re.fullmatch(r"\d+[)]\n", line) or line == "\n":
            person.clear()
            continue
        person.append(line[line.find(":") + 2 : -1])
        if len(person) == 6:
            if not correct_person(person):
                continue
            if person[1] not in names:
                names[person[1]] = 1
            else:
                names[person[1]] += 1
    print_result(names)


if __name__ == "__main__":
    main()
