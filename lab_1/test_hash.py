import unittest
import tempfile
import os
import sys

sys.path.append('..')
from hash import *


class TestHash(unittest.TestCase):
    """
    Класс для тестирования функций хеширования.
    Проверяет корректность работы всех основных функций модуля hash.py.
    """
    def test_hash_text(self):
        "Тестирует функцию hash_text()"
        h1 = hash_text("Hello")
        h2 = hash_text("Hello")
        h3 = hash_text("Hello!")
        
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertEqual(len(h1), 64)
    
    def test_hash_file(self):
        "Тестирует функцию hash_file()"
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello")
            tmp = f.name
        
        h_file = hash_file(tmp)
        h_text = hash_text("Hello")
        
        self.assertEqual(h_file, h_text)
        os.unlink(tmp)
    
    def test_save_load_hash(self):
        "Тестирует функции save_hash() и load_hash()"
        h = "abc123"
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            tmp = f.name
        
        self.assertTrue(save_hash(h, tmp))
        loaded = load_hash(tmp)
        self.assertEqual(loaded, h)
        os.unlink(tmp)
    
    def test_integrity(self):
        "Тестирует функцию check_integrity()"
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello")
            tmp = f.name
        
        h = hash_file(tmp)
        self.assertTrue(check_integrity(tmp, h))
        
        with open(tmp, 'a') as f:
            f.write("!")
        
        self.assertFalse(check_integrity(tmp, h))
        os.unlink(tmp)
    
    def test_avalanche(self):
        "Тестирует функцию avalanche()"
        h1, h2, diff, percent = avalanche("Hello", "Hello!")
        self.assertGreater(diff, 100)
        self.assertLess(percent, 80)


if __name__ == "__main__":
    unittest.main()