import argparse
import re
from datetime import datetime


def read_file(file_name: str) -> str:
    """Чтение файла."""
    try:
        with open(file_name, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError("Файл не найден") from exc


def parsing() -> str:
    """Передача аргументов через командную строку."""
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help="type filename")
    args = parser.parse_args()
    return str(args.file_name)


def is_date_valid(card: str) -> bool:
    """Проверка даты на валидность."""
    date_match = re.search(r"(\d{1,2})([-/.])(\d{1,2})\2(\d{4})", card)
    if date_match is None:
        return False
    day = date_match.group(1)
    month = date_match.group(3)
    year = date_match.group(4)
    try:
        parsed_date = datetime.strptime(f"{day}.{month}.{year}", "%d.%m.%Y")
    except ValueError:
        return False
    return parsed_date.year >= 1900


def get_cards(data: str) -> dict[int, str]:
    """Запись анкет в словарь."""
    cards_dict: dict[int, str] = {}
    cards = re.split(r"\d+\)\n", data)
    for i in range(1, len(cards)):
        if is_date_valid(cards[i]):
            cards_dict[i] = cards[i]
    return cards_dict


def get_age(cards: dict[int, str], index: int) -> datetime:
    """Парсинг и возврат даты в формате datetime."""
    date_match = re.search(r"(\d{1,2})([-/.])(\d{1,2})\2(\d{4})", cards[index])
    if date_match is None:
        raise ValueError(f"Дата не найдена в анкете {index}")
    day = date_match.group(1)
    month = date_match.group(3)
    year = date_match.group(4)
    return datetime.strptime(f"{day}.{month}.{year}", "%d.%m.%Y")


def get_oldest_and_youngest(cards: dict[int, str]) -> tuple[int, int, int, int]:
    """Поиск самого младшего и самого старшего человека."""
    if not cards:
        raise ValueError("Валидные анкеты не найдены")

    iterator = iter(cards)
    first_id = next(iterator)
    oldest_id = youngest_id = first_id
    oldest_age = youngest_age = get_age(cards, first_id)

    for i in iterator:
        age = get_age(cards, i)
        if age < oldest_age:
            oldest_age = age
            oldest_id = i
        if age > youngest_age:
            youngest_age = age
            youngest_id = i

    now = datetime.now()
    oldest = now.year - oldest_age.year
    if (now.month, now.day) < (oldest_age.month, oldest_age.day):
        oldest -= 1

    youngest = now.year - youngest_age.year
    if (now.month, now.day) < (youngest_age.month, youngest_age.day):
        youngest -= 1

    return oldest, oldest_id, youngest, youngest_id


def main() -> None:
    try:
        file_name = parsing()
        data = read_file(file_name)
        cards = get_cards(data)
        oldest, oldest_id, youngest, youngest_id = get_oldest_and_youngest(cards)
        print(
            f"Самый старый человек: \nВозраст: {oldest} \nАнкета: \n{cards[oldest_id]}"
        )
        print(
            f"Самый молодой человек: \nВозраст: {youngest} \n"
            f"Анкета: \n{cards[youngest_id]}"
        )
    except (FileNotFoundError, ValueError, OSError) as ex:
        print("Ошибка:", ex)


if __name__ == "__main__":
    main()
