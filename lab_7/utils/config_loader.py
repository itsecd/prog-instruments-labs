import json
import os
from typing import Dict, Any


class ConfigLoader:
    """Загрузчик конфигурации (Singleton)."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Создает единственный экземпляр класса Singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: str = "config.json") -> Dict[str, Any]:
        """
        Загружает конфигурацию из JSON файла.
        
        Args:
            config_path (str): путь к файлу
        
        Returns:
            Dict[str, Any]: словарь с настройками
        """
        match self._config:
            case None:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                return self._config
            case _:
                return self._config