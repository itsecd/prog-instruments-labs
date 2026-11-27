"""Filter questionnaires for women whose names begin with uppercase А."""

import argparse
import re
from collections.abc import Iterable

RECORD_HEADER = re.compile(r"\d+\)")


def is_name_correct(name: str) -> bool:
    """Check that the name starts with А and contains Cyrillic letters."""
    return re.fullmatch(r"А[а-яё]*", name) is not None


def is_female(gender: str) -> bool:
    """Recognize the female gender values used in the dataset."""
    return re.fullmatch(r"(Ж|ж|Женский|женский)", gender) is not None


def parse_fields(form: list[str]) -> dict[str, str]:
    """Extract questionnaire values by field name, not by line position."""
    fields: dict[str, str] = {}
    for line in form:
        label, separator, value = line.partition(":")
        if separator:
            fields[label.strip()] = value.strip()
    return fields


def form_is_correct(form: list[str]) -> bool:
    """Check whether a questionnaire matches the selection criteria."""
    fields = parse_fields(form)
    return is_name_correct(fields.get("Имя", "")) and is_female(fields.get("Пол", ""))


def parse_forms(lines: Iterable[str]) -> list[list[str]]:
    """Group field lines by numbered questionnaire headings."""
    forms: list[list[str]] = []
    current: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if RECORD_HEADER.fullmatch(line):
            if current:
                forms.append(current)
            current = []
        elif ":" in line:
            current.append(line)

    if current:
        forms.append(current)
    return forms


def read_forms(filename: str) -> list[list[str]]:
    """Read the source file without mixing file I/O and record parsing."""
    with open(filename, encoding="utf-8") as source:
        return parse_forms(source)


def save_forms(forms: list[list[str]], path: str = "filtered_forms.txt") -> None:
    """Save the selected questionnaires in the original text format."""
    with open(path, "w", encoding="utf-8") as destination:
        for form in forms:
            destination.write("\n".join(form) + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    forms = read_forms(args.filename)
    correct = [form for form in forms if form_is_correct(form)]
    print(len(correct))
    save_forms(correct)


if __name__ == "__main__":
    main()
