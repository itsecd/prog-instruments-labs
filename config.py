import json
from typing import Dict, Any


class Config:
    """Класс для работы с конфигурацией приложения"""

    def __init__(self, config_data: Dict[str, Any]):
        self.PATHS = config_data.get('paths', {})
        self.SETTINGS = config_data.get('settings', {})

    @classmethod
    def from_json(cls, file_path: str) -> 'Config':
        """Создает конфиг из JSON файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return cls(config_data)

    def get_path(self, key: str) -> str:
        """Возвращает путь по ключу"""
        return self.PATHS.get(key, '')

    def get_setting(self, key: str) -> Any:
        """Возвращает настройку по ключу"""
        return self.SETTINGS.get(key)