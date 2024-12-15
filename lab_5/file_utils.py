import json

from typing import Any, Dict, List, Tuple


def read_file(file_path: str) -> str:
    """
    Чтение содержимого файла и возвращение его в виде строки.
    
    Parameters
    ----------
    file_path : str
        Путь к файлу для чтения.
        
    Returns
    -------
    str
        Содержимое файла.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Ошибка: файл '{file_path}' не найден.")
        return ""
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")
        return ""


def write_text_to_file(file_path: str, text: str) -> None:
    """
    Запись текста в файл.
    
    Parameters
    ----------
    file_path : str
        Путь к файлу для записи.
    text : str
        Текст для записи в файл.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Произошла ошибка при записи в файл: {e}")


def write_dict_to_json(file_path: str, data: Dict[str, Any]) -> None:
    """
    Запись данных словаря в JSON файл.
    
    Parameters
    ----------
    file_path : str
        Путь к JSON файлу для записи.
    data : dict
        Данные словаря для записи в JSON файл.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Произошла ошибка при записи в JSON файл: {e}")


def write_decryption_key(key_file_path: str, decryption_mapping: Dict[str, str]) -> None:
    """
    Запись ключа расшифровки в JSON файл.
    
    Parameters
    ----------
    key_file_path : str
        Путь к файлу для записи ключа.
    decryption_mapping : dict
        Словарь сопоставления символов для расшифровки.
    """
    try:
        with open(key_file_path, 'w', encoding='utf-8') as key_file:
            json.dump(decryption_mapping, key_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Произошла ошибка при записи ключа в файл: {e}")


def save_to_json(data: Any, filename: str) -> None:
    """
    Сохраняет данные в формате JSON в файл.

    Parameters
    ----------
    data : Any
        Данные для сохранения в JSON формате.
    filename : str
        Путь к файлу для сохранения данных.

    Returns
    -------
    None
    """
    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден.")
        return ""
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")
        return ""


def save_to_text(sorted_freq: List[Tuple[str, int]], filename: str) -> None:
    """
    Сохраняет отсортированную частоту символов в текстовый файл.

    Parameters
    ----------
    sorted_freq : List[Tuple[str, int]]
        Отсортированный список кортежей, содержащих символы и их частоты.
    filename : str
        Путь к файлу для сохранения отсортированных данных.

    Returns
    -------
    None
    """
    try:
        with open(filename, 'w', encoding='UTF-8') as text_file:
            for char, frequency in sorted_freq:
                text_file.write(f"{char}: {frequency}\n")
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден.")
        return ""
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")
        return ""


def read_settings(settings_file_path: str) -> Tuple[str, str]:
    """
    Read encryption and decryption settings from a JSON file.
    Args:
        settings_file_path (str): Path to the settings JSON file.
    Returns:
        Tuple[str, str]: Tuple containing the crypt alphabet and normal alphabet.
    """
    try:
        with open(settings_file_path, 'r', encoding='UTF-8') as settings_file:
            settings = json.load(settings_file)
            return settings
    except FileNotFoundError:
        print(f"Ошибка: файл '{settings_file_path}' не найден.")
        return ""
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")
        return ""