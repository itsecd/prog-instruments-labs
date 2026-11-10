import unittest
from script import extract_phone_number, parse_profiles, find_duplicate_phones


class TestPhoneScript(unittest.TestCase):
    """Тесты для скрипта поиска дубликатов номеров телефонов."""

    def test_extract_phone_number(self):
        """Тестирование функции извлечения и нормализации номеров."""
        self.assertEqual(extract_phone_number("+7 927 345 67 89"), "+79273456789")
        self.assertEqual(extract_phone_number("8 (999) 456-78-90"), "+79994567890")
        self.assertIsNone(extract_phone_number("sid111@gmail.com"))

    def test_parse_profiles_and_duplicates(self):
        """Тестирование парсинга текста и поиска дубликатов."""
        text = """1)
Фамилия: Иванов
Имя: Алексей
Номер телефона или email: +7 927 345 67 89

2)
Фамилия: Петров
Имя: Иван
Номер телефона или email: 8 (927) 345-67-89
"""
        profiles = parse_profiles(text)
        self.assertEqual(len(profiles), 2)

        duplicates = find_duplicate_phones(profiles)
        self.assertEqual(len(duplicates), 1)
        self.assertIn("+79273456789", duplicates)
        self.assertEqual(len(duplicates["+79273456789"]), 2)


if __name__ == '__main__':
    unittest.main()