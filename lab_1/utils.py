import json
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    """
    Загружает JSON-конфигурацию из файла.
    """
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config not found: {path}") from e

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {path}") from e

    except PermissionError as e:
        raise PermissionError(f"Read permission denied: {path}") from e


def read_bytes(path: str) -> bytes:
    """
    Читает файл в бинарном режиме.
    """
    try:
        with open(path, "rb") as f:
            return f.read()

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {path}") from e

    except PermissionError as e:
        raise PermissionError(f"Read permission denied: {path}") from e

    except OSError as e:
        raise OSError(f"Failed to read file: {path}") from e


def write_bytes(path: str, data: bytes) -> None:
    """
    Записывает бинарные данные в файл.
    """
    try:
        with open(path, "wb") as f:
            f.write(data)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Invalid output path: {path}") from e

    except PermissionError as e:
        raise PermissionError(f"Write permission denied: {path}") from e

    except OSError as e:
        raise OSError(f"Failed to write file: {path}") from e
