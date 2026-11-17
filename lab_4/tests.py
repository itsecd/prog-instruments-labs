"""
tests.py - Юнит-тесты для системы аутентификации
"""
import unittest
import os
import tempfile
from auth_core import PasswordHasher, UserDatabase


class TestPasswordHasher(unittest.TestCase):
    """Тесты хеширования"""
    
    def test_hash_unsafe(self):
        """Тест хеширования без соли"""
        hash1 = PasswordHasher.hash_unsafe("password123")
        hash2 = PasswordHasher.hash_unsafe("password123")
        hash3 = PasswordHasher.hash_unsafe("different")
        
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertEqual(len(hash1), 64) 
    
    def test_hash_with_salt(self):
        """Тест хеширования с солью"""
        hash1, salt1 = PasswordHasher.hash_with_salt("password123")
        hash2, salt2 = PasswordHasher.hash_with_salt("password123")
        
        self.assertNotEqual(salt1, salt2)
        self.assertNotEqual(hash1, hash2)
        self.assertEqual(len(salt1), 32)  
    
    def test_verify_with_salt(self):
        """Тест верификации с солью"""
        password = "my_secret_pass"
        pwd_hash, salt = PasswordHasher.hash_with_salt(password)
        
        self.assertTrue(PasswordHasher.verify_with_salt(password, salt, pwd_hash))
        self.assertFalse(PasswordHasher.verify_with_salt("wrong", salt, pwd_hash))
    
    def test_verify_unsafe(self):
        """Тест верификации без соли"""
        password = "simple_pass"
        pwd_hash = PasswordHasher.hash_unsafe(password)
        
        self.assertTrue(PasswordHasher.verify_unsafe(password, pwd_hash))
        self.assertFalse(PasswordHasher.verify_unsafe("wrong", pwd_hash))


class TestUserDatabase(unittest.TestCase):
    """Тесты базы данных"""
    
    def setUp(self):
        """Создаём временный файл БД для тестов"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_db.close()
        self.db = UserDatabase(self.temp_db.name)
    
    def tearDown(self):
        """Удаляем временный файл"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_add_and_get_user(self):
        """Тест добавления и получения пользователя"""
        result = self.db.add_user("alice", "hash123", "salt456", "SHA-256 with salt")
        self.assertTrue(result)
        
        user = self.db.get_user("alice")
        self.assertIsNotNone(user)
        self.assertEqual(user["hash"], "hash123")
        self.assertEqual(user["salt"], "salt456")
    
    def test_duplicate_user(self):
        """Тест запрета дубликатов"""
        self.db.add_user("bob", "hash1", "salt1")
        result = self.db.add_user("bob", "hash2", "salt2")
        self.assertFalse(result)
    
    def test_user_exists(self):
        """Тест проверки существования пользователя"""
        self.assertFalse(self.db.user_exists("charlie"))
        self.db.add_user("charlie", "hash", "salt")
        self.assertTrue(self.db.user_exists("charlie"))
    
    def test_get_unsafe_users(self):
        """Тест поиска уязвимых пользователей"""
        self.db.add_user("safe_user", "hash1", "salt1", "with salt")
        self.db.add_user("unsafe_user1", "hash2", None, "no salt")
        self.db.add_user("unsafe_user2", "hash3", None, "no salt")
        
        unsafe = self.db.get_unsafe_users()
        self.assertEqual(len(unsafe), 2)
        self.assertIn("unsafe_user1", unsafe)
        self.assertIn("unsafe_user2", unsafe)
        self.assertNotIn("safe_user", unsafe)
    
    def test_delete_all(self):
        """Тест очистки базы"""
        self.db.add_user("user1", "hash1", "salt1")
        self.db.add_user("user2", "hash2", "salt2")
        self.assertEqual(len(self.db.get_all_users()), 2)
        
        self.db.delete_all()
        self.assertEqual(len(self.db.get_all_users()), 0)
    
    def test_save_and_load(self):
        """Тест сохранения и загрузки"""
        self.db.add_user("persistent", "hash_val", "salt_val")
        self.db.save()
        
        new_db = UserDatabase(self.temp_db.name)
        user = new_db.get_user("persistent")
        self.assertIsNotNone(user)
        self.assertEqual(user["hash"], "hash_val")


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""
    
    def test_full_auth_flow_with_salt(self):
        """Полный цикл регистрации и авторизации с солью"""
        db = UserDatabase("test_integration.json")
        
        db.delete_all()
        
        password = "secure_pass_123"
        pwd_hash, salt = PasswordHasher.hash_with_salt(password)
        db.add_user("testuser", pwd_hash, salt, "SHA-256 with salt")
        
        self.assertTrue(db.user_exists("testuser"))
        
        user = db.get_user("testuser")
        is_valid = PasswordHasher.verify_with_salt(password, user["salt"], user["hash"])
        self.assertTrue(is_valid)
        
        is_valid_wrong = PasswordHasher.verify_with_salt("wrong", user["salt"], user["hash"])
        self.assertFalse(is_valid_wrong)
        
        db.delete_all()
        os.unlink("test_integration.json")


if __name__ == "__main__":
    unittest.main()