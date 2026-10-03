import hashlib


def hash_file(filepath):
    """
    Вычисляет SHA-256 хеш файла.
    Аргументы: filepath путь к файлу
    Возвращает: строку с хешем (64 символа) или None, если файл не найден
    """
    try:
        with open(filepath, "rb") as f:
            data = f.read()
            return hashlib.sha256(data).hexdigest()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def hash_text(text):
    """
    Вычисляет SHA-256 хеш строки текста.
    Аргументы: text строка текста
    Возвращает: строку с хешем (64 символа)
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_hash(hash, filepath):
    """
    Сохраняет хеш в файл.
    Аргументы: hash_value строка с хешем, filepath: путь к файлу
    Возвращает: True при успехе, False при ошибке
    """
    try:
        with open(filepath, "w") as f:
            f.write(hash)
        return True
    except (FileNotFoundError, PermissionError, OSError):
        return False


def load_hash(filepath):
    """
    Загружает хеш из файла.
    Аргументы: filepath путь к файлу
    Возвращает: строку с хешем или None, если файл не найден
    """
    try:
        with open(filepath) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def check_integrity(filepath, expected_hash):
    """
    Проверяет соответствие текущего хеша файла с ожидаемым.
    Аргументы: filepath путь к файлу, expected_hash ожидаемый хеш
    Возвращает: True если хеши совпадают, False если нет или ошибка
    """
    actual_hash = hash_file(filepath)
    if actual_hash is None:
        return False
    return actual_hash == expected_hash


def avalanche(text1, text2):
    """
    Показывает как меняется хеш при малом изменении текста (лавинный эффект).
    Аргументы: text1 первый текст, text2 второй текст
    Возвращает: хеш1, хеш2, количество отличающихся бит, процент
    """
    hash1 = hash_text(text1)
    hash2 = hash_text(text2)

    bits1 = bin(int(hash1, 16))[2:].zfill(256)
    bits2 = bin(int(hash2, 16))[2:].zfill(256)
    diff = sum(1 for i in range(256) if bits1[i] != bits2[i])
    percent = (diff / 256) * 100

    return hash1, hash2, diff, percent
