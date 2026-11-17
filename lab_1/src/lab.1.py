def process_and_sort_names(filename: str) -> list[str]:
    """
    Читает файл с данными, сортирует записи и преобразует к формату 'Фамилия И.'

    Args:
        filename (str): имя файла с данными

    Returns:
        list: отсортированный список имен в формате 'Фамилия И.'
    """
    result = []
    result: list[str]

    with open(filename, encoding="utf-8") as file:
        content = file.read()

    # Разделяем на отдельные записи
    records = content.split("\n\n")

    for record in records:
        if not record.strip():
            continue

        lines = record.strip().split("\n")
        surname = ""
        name = ""

        for line in lines:
            if line.startswith("Фамилия:"):
                surname = line.replace("Фамилия:", "").strip()
            elif line.startswith("Имя:"):
                name = line.replace("Имя:", "").strip()

        # Приводим к правильному регистру
        if surname:
            surname = surname.capitalize()
        if name:
            name = name.capitalize()

        # Форматируем в "Фамилия И."
        if surname and name:
            formatted_name = f"{surname} {name[0]}."
            result.append(formatted_name)

    # Сортируем по фамилии, затем по имени
    result.sort()

    return result


# Пример использования
if __name__ == "__main__":
    sorted_names = process_and_sort_names("data.txt")

    print("Отсортированный список имен в формате 'Фамилия И.':")
    print("-" * 40)
    for i, name in enumerate(sorted_names, 1):
        print(f"{i:3}. {name}")
