import os
from typing import Dict, Any, Union


class FileManager:
    """Управление файловыми операциями."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует менеджер файлов и создает директорию для ключей.
        
        Args:
            config (Dict[str, Any]): конфигурация
        """
        paths = config.get('paths')
        if paths is None:
            raise KeyError("Секция 'paths' не найдена")
        
        self._keys_dir = paths.get('keys_directory')
        if self._keys_dir is None:
            raise KeyError("Параметр 'keys_directory' не найден")
        
        os.makedirs(self._keys_dir, exist_ok=True)
        self._paths = paths
    
    def read_file(self, filepath: str, binary: bool = False) -> Union[str, bytes]:
        """
        Читает содержимое файла в текстовом или бинарном режиме.
        
        Args:
            filepath (str): путь к файлу
            binary (bool): бинарный режим
        
        Returns:
            Union[str, bytes]: содержимое файла
        """
        if not filepath:
            raise ValueError("Путь к файлу не может быть пустым")
        
        mode = 'rb' if binary else 'r'
        with open(filepath, mode) as f:
            return f.read()
    
    def write_file(self, filepath: str, data: Union[str, bytes], binary: bool = False) -> None:
        """
        Записывает данные в файл в текстовом или бинарном режиме.
        
        Args:
            filepath (str): путь к файлу
            data (Union[str, bytes]): данные
            binary (bool): бинарный режим
        """
        if not filepath:
             raise ValueError("Путь к файлу не может быть пустым")
        
        if data is None:
            raise ValueError("Нет данных для записи")
        
        directory = os.path.dirname(filepath)
        if not directory:
            os.makedirs('.', exist_ok=True)
        else:
            os.makedirs(directory, exist_ok=True)
        
        mode = 'wb' if binary else 'w'
        with open(filepath, mode) as f:
            f.write(data)
    
    def file_exists(self, filepath: str) -> bool:
        """
        Проверяет существование файла по указанному пути.
        
        Args:
            filepath (str): путь к файлу
        
        Returns:
            bool: True если существует
        """
        if not filepath:
            return False
        return os.path.exists(filepath)
    
    def get_path(self, key: str) -> str:
        """
        Возвращает полный путь к файлу, объединяя keys_directory с именем файла.
        
        Args:
            key (str): ключ в секции 'paths'
        
        Returns:
            str: полный путь
        """
        filename = self._paths.get(key)
        if filename is None:
            raise KeyError(f"Параметр '{key}' не найден")
        if not filename:
            raise ValueError(f"Путь для ключа '{key}' пуст")
        
        return os.path.join(self._keys_dir, filename)