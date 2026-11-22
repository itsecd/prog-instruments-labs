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
        match paths:
            case None:
                raise KeyError("Секция 'paths' не найдена")
        
        self._keys_dir = paths.get('keys_directory')
        match self._keys_dir:
            case None:
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
        match filepath:
            case '':
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
        match filepath:
            case '':
                raise ValueError("Путь к файлу не может быть пустым")
        
        match data:
            case None:
                raise ValueError("Нет данных для записи")
        
        directory = os.path.dirname(filepath)
        match directory:
            case '':
                os.makedirs('.', exist_ok=True)
            case _:
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
        match filepath:
            case '':
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
        match filename:
            case None:
                raise KeyError(f"Параметр '{key}' не найден")
            case '':
                raise ValueError(f"Путь для ключа '{key}' пуст")
        
        return os.path.join(self._keys_dir, filename)