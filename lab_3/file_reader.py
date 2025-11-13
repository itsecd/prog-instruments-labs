import chardet

"""
Чтение csv-файла с проверкой кодировки
"""


def read_csv_file(file_path: str) -> list:
    """
    Читает CSV файл с проверкой кодировки

    Returns:
        list: список строк файла
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)

    encoding = result['encoding']
    print(f"Кодировка файла: {encoding}")

    with open(file_path, 'r', encoding=encoding) as f:
        lines = f.readlines()

    print(f"Файл успешно прочитан, строк: {len(lines)}")
    return lines
