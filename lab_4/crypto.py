import hashlib
from tqdm import tqdm
from read_write_file import save_hash_to_file, load_hash_from_file

def calculate_sha256(file_path : str) -> str:
    """
    Вычисляет SHA-256 хеш файла
    :param file_path: путь к файлу с данными
    :return: вычисленный хеш файла
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    except PermissionError:
        raise PermissionError(f"Нет прав для чтения файла: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Ошибка при чтении файла: {e}")

def save_file_hash(file_path : str) -> str:
    """
    Создание пути для хеш файла и его сохранение
    :param file_path: путь к исходному файлу
    :return: путь к хеш файлу
    """
    current_hash = calculate_sha256(file_path)
    save_path = f"{file_path}.sha256"
    save_hash_to_file(current_hash, save_path)
    return save_path

def verify_file_hash(file_path : str, hash_file_path : str)-> tuple[bool, str, str]:
    """
    Сопоставление хешей
    :param file_path: путь к проверяемому файлу
    :param hash_file_path: путь к хеш файлу
    :return: да или нет, текущий хеш, эталонный хеш
    """
    current_hash = calculate_sha256(file_path)
    saved_hash = load_hash_from_file(hash_file_path)
    return current_hash == saved_hash, current_hash, saved_hash


def find_partial_collision(target_prefix : str, max_iterations=10000000, out_file=None)-> tuple[str, str]:
    """
    Поиск коллизии
    :param target_prefix: искомый хеш
    :param max_iterations: максимальное количество попыток подбора
    :param out_file: Поток для визуализации процесса подбора коллизии при помощи tqdm
    :return: подобранный текст и полученный хеш
    """
    for i in tqdm(range(max_iterations), desc=f"Поиск '{target_prefix}'", file=out_file):
        candidate = f"{i}"
        current_hash = hashlib.sha256(candidate.encode()).hexdigest()
        if current_hash.startswith(target_prefix):
            return candidate, current_hash
    return None, None



if __name__ == "__main__":
    test_file = "crypto.py"

    h = calculate_sha256(test_file)
    print("Хеш модуля:", h)

    saved = save_file_hash(test_file)
    print("Хеш сохранен в:", saved)

    c, k = find_partial_collision("abc123", max_iterations=10000)
    print(f"Результат: {c}, {k}")