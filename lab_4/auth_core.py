"""
auth_core.py - Ядро системы аутентификации с хешированием паролей
"""
import hashlib
import os
import json
import time
from typing import Dict, Optional, Tuple


class AuthError(Exception):
    """Базовое исключение для ошибок аутентификации"""
    pass


class UserExistsError(AuthError):
    pass


class UserNotFoundError(AuthError):
    pass


class InvalidPasswordError(AuthError):
    pass


class PasswordHasher:
    """Класс для хеширования паролей с солью и без"""
    
    @staticmethod
    def hash_unsafe(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def hash_with_salt(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        if salt is None:
            salt = os.urandom(16).hex()
        salted_password = password + salt
        password_hash = hashlib.sha256(salted_password.encode()).hexdigest()
        return password_hash, salt
    
    @staticmethod
    def verify_with_salt(password: str, salt: str, stored_hash: str) -> bool:
        computed_hash, _ = PasswordHasher.hash_with_salt(password, salt)
        return computed_hash == stored_hash
    
    @staticmethod
    def verify_unsafe(password: str, stored_hash: str) -> bool:
        return PasswordHasher.hash_unsafe(password) == stored_hash


class UserDatabase:
    """Управление базой пользователей"""
    
    def __init__(self, db_file: str = "users.json"):
        self.db_file = db_file
        self.users: Dict = {}
        self.load()
    
    def load(self) -> None:
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    self.users = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # В идеале здесь должен быть logging, но для лабы оставим print
            print(f"Ошибка загрузки БД: {e}")
            self.users = {}
    
    def save(self) -> None:
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.users, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise IOError(f"Не удалось сохранить БД: {e}")
    
    def add_user(self, username: str, password_hash: str, salt: Optional[str] = None, 
                 method: str = "SHA-256 with salt") -> None:
        if username in self.users:
            raise UserExistsError(f"Пользователь '{username}' уже существует")
        self.users[username] = {
            "hash": password_hash,
            "salt": salt,
            "method": method,
            "created_at": time.time()
        }
        self.save()
    
    def get_user(self, username: str) -> Dict:
        user = self.users.get(username)
        if not user:
            raise UserNotFoundError(f"Пользователь '{username}' не найден")
        return user
    
    def user_exists(self, username: str) -> bool:
        return username in self.users
    
    def delete_all(self) -> None:
        self.users = {}
        self.save()
    
    def get_all_users(self) -> Dict:
        return self.users
    
    def get_unsafe_users(self) -> list:
        return [u for u, data in self.users.items() if not data.get("salt")]


class AuthService:
    """Бизнес-логика аутентификации (отделена от UI и CLI)"""
    
    def __init__(self, db: UserDatabase):
        self.db = db

    def register(self, username: str, password: str, use_salt: bool = True) -> str:
        """Регистрация пользователя. Возвращает строку с методом."""
        if use_salt:
            pwd_hash, salt = PasswordHasher.hash_with_salt(password)
            method = "SHA-256 with salt"
        else:
            pwd_hash = PasswordHasher.hash_unsafe(password)
            salt = None
            method = "SHA-256 without salt"
        
        self.db.add_user(username, pwd_hash, salt, method)
        return method

    def login(self, username: str, password: str, use_salt: bool = True) -> None:
        """Авторизация. Выбрасывает исключение при неудаче."""
        user = self.db.get_user(username)
        
        if use_salt:
            if not user.get("salt"):
                raise AuthError("Этот пользователь зарегистрирован без соли")
            success = PasswordHasher.verify_with_salt(password, user["salt"], user["hash"])
        else:
            success = PasswordHasher.verify_unsafe(password, user["hash"])
            
        if not success:
            raise InvalidPasswordError("Неверный пароль")


class CollisionDemo:
    """Демонстрация поиска коллизий (без привязки к UI/CLI)"""
    
    @staticmethod
    def find_collision(target_hash: str, max_attempts: int = 1000000) -> Optional[str]:
        """
        Поиск коллизии. Возвращает найденный пароль или None.
        Прогресс-бар (tqdm) перенесен в CLI.
        """
        for i in range(max_attempts):
            test_password = f"password_{i}"
            test_hash = PasswordHasher.hash_unsafe(test_password)
            
            if test_hash == target_hash:
                return test_password
                
        return None