import json
from typing import List, Dict, Optional

class User:
    def __init__(self, user_id: int, username: str, email: str):
        self.user_id = user_id
        self.username = username
        self.email = email

    def to_dict(self) -> Dict[str, str]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email
        }

class UserService:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.users: List[User] = self.load_users()

    def load_users(self) -> List[User]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return [User(**user) for user in data]
        except FileNotFoundError:
            print("Файл не найден. Начинаем с пустого списка пользователей.")
            return []
        except json.JSONDecodeError:
            print("Ошибка при декодировании JSON. Начинаем с пустого списка пользователей.")
            return []

    def save_users(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump([user.to_dict() for user in self.users], file, ensure_ascii=False, indent=4)

    def add_user(self, username: str, email: str) -> None:
        if self.is_username_taken(username):
            raise ValueError(f"Имя пользователя {username} уже занято.")
        
        new_id = self.generate_user_id()
        new_user = User(new_id, username, email)
        self.users.append(new_user)
        self.save_users()

    def is_username_taken(self, username: str) -> bool:
        return any(user.username == username for user in self.users)

    def generate_user_id(self) -> int:
        return max((user.user_id for user in self.users), default=0) + 1

    def get_user(self, user_id: int) -> Optional[User]:
        for user in self.users:
            if user.user_id == user_id:
                return user
        return None

    def remove_user(self, user_id: int) -> None:
        user_to_remove = self.get_user(user_id)
        if user_to_remove:
            self.users.remove(user_to_remove)
            self.save_users()
        else:
            raise ValueError(f"Пользователь с id: {user_id} не найден.")

    def update_user(self, user_id: int, username: Optional[str] = None, 
                    email: Optional[str] = None) -> None:
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"Пользователь с ID: {user_id} не найден.")
        
        if username and self.is_username_taken(username):
            raise ValueError(f"Имя пользователя: {username} уже занято.")
        
        if username:
            user.username = username
        if email:
            user.email = email
        
        self.save_users()


if __name__ == "__main__":
    user_service = UserService("lab_5/users.json")
    user_service.add_user("john_doe", "john@example.com")
    user_service.add_user("jane_doe", "jane@example.com")
    print(user_service.get_user(1).to_dict())
    user_service.update_user(1, email="john_new@example.com")
    user_service.remove_user(2)
