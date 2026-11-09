#!/usr/bin/env python3

import re
import argparse


def format_questionnaires(data: list[str]) -> str:
    """
    Форматирует анкеты с нумерацией.
    """
    return "".join(
        f"{i})\n{dat}\n\n"
        for i, dat in enumerate(data, 1)
    )


def read_file(filename: str) -> str:
    """Читает содержимое файла."""
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def write_file(filename: str, data: str) -> None:
    """
    Записывает готовый текст в файл.
    """
    with open(filename, "w", encoding="utf-8") as file:
        file.write(data)


def args_parse() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    """
    parser = argparse.ArgumentParser(
        prog="Data parser", description="Parsing data from file"
    )
    parser.add_argument("-f", "--file", type=str, help="Parse file")
    parser.add_argument("-o", "--output", type=str, help="Output file")
    return parser.parse_args()


def extract_questionnaire(part: str) -> dict[str, str] | None:
    """
    Извлекает фамилию, имя и текст анкеты.
    """
    last_name_match = re.search(r"Фамилия:\s*([^\n]+)", part)
    first_name_match = re.search(r"Имя:\s*([^\n]+)", part)

    if not last_name_match or not first_name_match:
        return None

    return {
        "last_name": last_name_match.group(1).strip(),
        "first_name": first_name_match.group(1).strip(),
        "text": part,
    }


def parse_data(data: str) -> list[dict[str, str]]:
    """
    Разделяет текст на анкеты и извлекает данные.
    """
    parts = [p.strip() for p in re.split(r"\n\d+\)\s*", data) if p.strip()]

    questionnaires = []

    for part in parts:
        questionnaire = extract_questionnaire(part)

        if questionnaire is not None:
            questionnaires.append(questionnaire)

    return questionnaires


def sort_questionnaires(
    questionnaires: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Сортирует анкеты по фамилии и имени.
    """
    return sorted(
        questionnaires,
        key=lambda q: (
            q["last_name"].lower(),
            q["first_name"].lower(),
        ),
    )


def main():
    args = args_parse()

    if not args.file:
        print("Please specify input file with -f option")
        return

    try:
        data = read_file(args.file)
    except OSError as error:
        print(f"File error: {error}")
        return

    if not data:
        return

    questionnaires = parse_data(data)
    sorted_questionnaires = sort_questionnaires(questionnaires)

    sorted_data = [q["text"] for q in sorted_questionnaires]
    result = format_questionnaires(sorted_data)

    if args.output:
        write_file(args.output, result)
        print(f"Sorted data written to {args.output}")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
