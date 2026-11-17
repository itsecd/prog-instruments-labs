"""
tests.py - Юнит-тесты для системы аутентификации
"""
import unittest
import os
import tempfile
from auth_core import PasswordHasher, UserDatabase, AuthService, UserExistsError, InvalidPasswordError


class TestPasswordHasher(unittest.TestCase):
    """Тесты хеширования"""
    
    def test_hash_unsafe(self):
        hash1 = PasswordHasher.hash_unsafe("password123")
        hash2 = PasswordHasher.hash_unsafe("password123")
        hash3 = PasswordHasher.hash_unsafe("different")
        
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertEqual(len(hash1), 64) 
    
    def test_hash_with_salt(self):
        hash1, salt1 = PasswordHasher.hash_with_salt("password123")
        hash2, salt2 = PasswordHasher.hash_with_salt("password123")
        
        self.assertNotEqual(salt1, salt2)
        self.assertNotEqual(hash1, hash2)
        self.assertEqual(len(salt1), 32)  
    
    def test_verify_with_salt(self):
        password = "my_secret_pass"
        pwd_hash, salt = PasswordHasher.hash_with_salt(password)
        
        self.assertTrue(PasswordHasher.verify_with_salt(password, salt, pwd_hash))
        self.assertFalse(PasswordHasher.verify_with_salt("wrong", salt, pwd_hash))
    
    def test_verify_unsafe(self):
        password = "simple_pass"
        pwd_hash = PasswordHasher.hash_unsafe(password)
        
        self.assertTrue(PasswordHasher.verify_unsafe(password, pwd_hash))
        self.assertFalse(PasswordHasher.verify_unsafe("wrong", pwd_hash))


class TestUserDatabase(unittest.TestCase):
    """Тесты базы данных"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_db.close()
        self.db = UserDatabase(self.temp_db.name)
    
    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_add_and_get_user(self):
        self.db.add_user("alice", "hash123", "salt456", "SHA-256 with salt")
        user = self.db.get_user("alice")
        self.assertIsNotNone(user)
        self.assertEqual(user["hash"], "hash123")
        self.assertEqual(user["salt"], "salt456")
    
    def test_duplicate_user(self):
        self.db.add_user("bob", "hash1", "salt1")
        with self.assertRaises(UserExistsError):
            self.db.add_user("bob", "hash2", "salt2")
    
    def test_get_unsafe_users(self):
        self.db.add_user("safe_user", "hash1", "salt1", "with salt")
        self.db.add_user("unsafe_user1", "hash2", None, "no salt")
        self.db.add_user("unsafe_user2", "hash3", None, "no salt")
        
        unsafe = self.db.get_unsafe_users()
        self.assertEqual(len(unsafe), 2)
        self.assertIn("unsafe_user1", unsafe)
        self.assertIn("unsafe_user2", unsafe)
        self.assertNotIn("safe_user", unsafe)
    
    def test_save_and_load(self):
        self.db.add_user("persistent", "hash_val", "salt_val")
        self.db.save()
        
        new_db = UserDatabase(self.temp_db.name)
        user = new_db.get_user("persistent")
        self.assertIsNotNone(user)
        self.assertEqual(user["hash"], "hash_val")


class TestAuthService(unittest.TestCase):
    """Тесты бизнес-логики AuthService"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.temp_db.close()
        self.db = UserDatabase(self.temp_db.name)
        self.service = AuthService(self.db)
        
    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_full_auth_flow_with_salt(self):
        password = "secure_pass_123"
        self.service.register("testuser", password, use_salt=True)
        
        self.assertTrue(self.db.user_exists("testuser"))
        
        # Успешный вход
        self.service.login("testuser", password, use_salt=True)
        
        # Неуспешный вход
        with self.assertRaises(InvalidPasswordError):
            self.service.login("testuser", "wrong", use_salt=True)


if __name__ == "__main__":
    unittest.main()