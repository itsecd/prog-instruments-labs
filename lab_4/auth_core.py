"""
auth_core.py - Ядро системы аутентификации с хешированием паролей
"""
import hashlib
import os
import json
import time
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class PasswordHasher:
    """Класс для хеширования паролей с солью и без"""
    
    @staticmethod
    def hash_unsafe(password: str) -> str:
        """SHA-256 без соли (уязвимый метод)"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def hash_with_salt(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """SHA-256 с солью (безопасный метод)"""
        if salt is None:
            salt = os.urandom(16).hex()
        salted_password = password + salt
        password_hash = hashlib.sha256(salted_password.encode()).hexdigest()
        return password_hash, salt
    
    @staticmethod
    def verify_with_salt(password: str, salt: str, stored_hash: str) -> bool:
        """Проверка пароля с солью"""
        computed_hash, _ = PasswordHasher.hash_with_salt(password, salt)
        return computed_hash == stored_hash
    
    @staticmethod
    def verify_unsafe(password: str, stored_hash: str) -> bool:
        """Проверка пароля без соли"""
        return PasswordHasher.hash_unsafe(password) == stored_hash


class UserDatabase:
    """Управление базой пользователей"""
    
    def __init__(self, db_file: str = "users.json"):
        self.db_file = db_file
        self.users: Dict = {}
        self.load()
    
    def load(self) -> None:
        """Загрузить пользователей из файла"""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    self.users = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка загрузки БД: {e}")
            self.users = {}
    
    def save(self) -> None:
        """Сохранить пользователей в файл"""
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.users, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise IOError(f"Не удалось сохранить БД: {e}")
    
    def add_user(self, username: str, password_hash: str, salt: Optional[str] = None, 
                 method: str = "SHA-256 with salt") -> bool:
        """Добавить пользователя"""
        if username in self.users:
            return False
        self.users[username] = {
            "hash": password_hash,
            "salt": salt,
            "method": method,
            "created_at": time.time()
        }
        self.save()
        return True
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Получить данные пользователя"""
        return self.users.get(username)
    
    def user_exists(self, username: str) -> bool:
        return username in self.users
    
    def delete_all(self) -> None:
        """Очистить базу"""
        self.users = {}
        self.save()
    
    def get_all_users(self) -> Dict:
        return self.users
    
    def get_unsafe_users(self) -> list:
        """Получить список пользователей без соли (уязвимых)"""
        return [u for u, data in self.users.items() if not data.get("salt")]


class CollisionDemo:
    """Демонстрация поиска коллизий с визуализацией """
    
    @staticmethod
    def find_collision_demo(target_hash: str, max_attempts: int = 1000000) -> Optional[str]:
        """
        Демонстрация поиска коллизии
        """
        print(f"\n Поиск пароля для хеша: {target_hash[:16]}...")
        print(f"   (демонстрация перебора с tqdm, максимально {max_attempts} попыток)")
        
        with tqdm(total=max_attempts, desc="Перебор паролей", unit="попытка") as pbar:
            for i in range(max_attempts):
                test_password = f"password_{i}"
                test_hash = PasswordHasher.hash_unsafe(test_password)
                
                pbar.update(1)
                
                if test_hash == target_hash:
                    pbar.set_description(" Коллизия найдена!")
                    return test_password
                
                if i % 10000 == 0:
                    pbar.set_postfix({"Текущий": test_password[:10]})
        
        return None