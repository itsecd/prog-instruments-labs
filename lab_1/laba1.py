"""Поиск людей по имени в анкетах."""

import argparse

FIELD_NAMES = {
    "фамилия": "surname",
    "имя": "name",
    "пол": "gender",
    "дата рождения": "birth_date",
    "номер телефона или email": "contact",
    "город": "city",
}


def parse_arguments() -> argparse.Namespace:
    """Читает аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Поиск людей по имени в анкетах")
    parser.add_argument("file", help="Путь к файлу с анкетами")
    return parser.parse_args()


def read_profiles(filename: str) -> list[str]:
    """Читает анкеты из файла."""
    with open(filename, encoding="utf-8") as file:
        content = file.read()

    return [block.strip() for block in content.split("\n\n") if block.strip()]


def parse_profile(record: str) -> dict[str, str]:
    """Разбирает одну анкету."""
    profile: dict[str, str] = {}

    for line in record.splitlines():
        if ":" not in line:
            continue

        label, value = line.split(":", 1)
        field = FIELD_NAMES.get(label.strip().lower())

        if field is not None:
            profile[field] = value.strip()

    return profile


def find_people_by_name(records: list[str], target_name: str) -> list[dict[str, str]]:
    """Ищет анкеты по имени."""
    matches: list[dict[str, str]] = []

    for record in records:
        profile = parse_profile(record)
        name = profile.get("name", "")

        if name.lower() == target_name.lower():
            matches.append(profile)

    return matches


def save_profiles(profiles: list[dict[str, str]], output_filename: str) -> None:
    """Сохраняет найденные анкеты в файл."""
    with open(output_filename, "w", encoding="utf-8") as file:
        for profile in profiles:
            file.write(f"Фамилия: {profile.get('surname', '')}\n")
            file.write(f"Имя: {profile.get('name', '')}\n")
            file.write(f"Пол: {profile.get('gender', '')}\n")
            file.write(f"Дата рождения: {profile.get('birth_date', '')}\n")
            file.write(f"Номер телефона или email: {profile.get('contact', '')}\n")
            file.write(f"Город: {profile.get('city', '')}\n")
            file.write("\n")


def main() -> None:
    """Запускает программу."""
    args = parse_arguments()

    try:
        records = read_profiles(args.file)
    except OSError as error:
        print(f"Ошибка при чтении файла: {error}")
        return

    if not records:
        print("Не удалось загрузить анкеты из файла")
        return

    print(f"Загружено анкет: {len(records)}")
    target_name = input("Введите имя для поиска: ").strip()

    if not target_name:
        print("Имя не может быть пустым")
        return

    matches = find_people_by_name(records, target_name)
    print("\nРезультаты поиска:")
    print(f"Людей с именем '{target_name}': {len(matches)}")

    if not matches:
        print("Люди с таким именем не найдены.")
        return

    output_filename = f"people_{target_name}.txt"

    try:
        save_profiles(matches, output_filename)
    except OSError as error:
        print(f"Ошибка при сохранении файла: {error}")
        return

    print(f"Найденные анкеты сохранены в файл: {output_filename}")


if __name__ == "__main__":
    main()
