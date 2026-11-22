import json


def load_config(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка чтения JSON: {e}")


def save_config(path, config):
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4, ensure_ascii=False)
    except OSError as e:
        raise OSError(f"Ошибка сохранения конфигурации: {e}")