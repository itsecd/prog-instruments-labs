def save_hash_to_file(hash_value : str, output_path : str) -> None:
    """
    Сохраняет строку хеша в указанный файл
    :param hash_value: хеш
    :param output_path: путь для сохранения хеша в файл
    :return: ничего
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(hash_value)
    except Exception as e:
        raise RuntimeError(f"Не удалось сохранить хеш в файл: {e}")

def load_hash_from_file(hash_file_path : str) -> str:
    """
    Считывает сохраненный хеш из файла
    :param hash_file_path: путь к файлу с хешем
    :return: считанный хеш
    """
    try:
        with open(hash_file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать хеш из файла: {e}")


if __name__ == "__main__":
    save_hash_to_file("abcd1236fvd", "data.sha256")
    print("Проверка чтения:", load_hash_from_file("data.sha256"))