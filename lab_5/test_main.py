import pytest

from unittest.mock import patch, mock_open
from main import User, UserService


def test_add_user():
    service = UserService("test_users.json")
    service.users = []
    service.add_user("john_doe", "john@example.com")

    assert len(service.users) == 1
    assert service.users[0].username == "john_doe"
    assert service.users[0].email == "john@example.com"


def test_get_user():
    service = UserService("test_users.json")
    service.users = [User(1, "john_doe", "john@example.com")]

    user = service.get_user(1)
    
    assert user is not None
    assert user.username == "john_doe"


def test_remove_user():
    service = UserService("test_users.json")
    service.users = [User(1, "john_doe", "john@example.com")]

    service.remove_user(1)
    
    assert len(service.users) == 0


def test_update_user():
    service = UserService("test_users.json")
    service.users = [User(1, "john_doe", "john@example.com")]

    service.update_user(1, email="john_new@example.com")
    
    assert service.users[0].email == "john_new@example.com"


@pytest.mark.parametrize("username, expected", [
    ("john_doe", True),
    ("jane_doe", False),
])
def test_is_username_taken(username, expected):
    service = UserService("test_users.json")
    service.users = [User(1, "john_doe", "john@example.com")]

    assert service.is_username_taken(username) == expected


def test_load_users_with_mock():
    mock_data = '[{"user_id": 1, "username": "john_doe", "email": "john@example.com"}]'
    
    with patch("builtins.open", mock_open(read_data = mock_data)):
        service = UserService("test_users.json")
        
        assert len(service.users) == 1
        assert service.users[0].username == "john_doe"


def test_remove_nonexistent_user():
    service = UserService("test_users.json")
    service.users = [User(1, "john_doe", "john@example.com")]

    with pytest.raises(ValueError, match = "Пользователь с id: 2 не найден."):
        service.remove_user(2)


def test_update_nonexistent_user():
    service = UserService("test_users.json")
    service.users = [User(1, "john_doe", "john@example.com")]

    with pytest.raises(ValueError, match = "Пользователь с ID: 2 не найден."):
        service.update_user(2, email = "john_new@example.com")
