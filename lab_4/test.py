import unittest
import os
import hashlib
from crypto import calculate_sha256, save_file_hash, verify_file_hash


class TestHashIntegrity(unittest.TestCase):
    def setUp(self)->None:
        """
        Подготовка временного окружения для каждого теста
        :return: ничего
        """
        self.test_file = "temp_test_file.txt"
        self.hash_file = "temp_test_file.txt.sha256"
        self.content = b"Hello, World!"

        with open(self.test_file, "wb") as f:
            f.write(self.content)

        self.expected_hash = hashlib.sha256(self.content).hexdigest()

    def tearDown(self):
        """
        Очистка временных файлов после теста
        :return: Ничего
        """
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.hash_file):
            os.remove(self.hash_file)

    def test_calculate_hash_correct(self):
        """
        Проверка базового вычисления хеша
        :return: ничего
        """
        result_hash = calculate_sha256(self.test_file)
        self.assertEqual(result_hash, self.expected_hash)

    def test_file_not_found_exception(self):
        """
        Проверка правильной обработки критической ошибки
        :return: ничего
        """
        with self.assertRaises(FileNotFoundError):
            calculate_sha256("non_existent_file.xyz")

    def test_save_file_hash_creates_file(self):
        """
        Тест новой функции: создание .sha256 файла рядом с исходным
        :return: ничего
        """
        created_path = save_file_hash(self.test_file)
        self.assertTrue(os.path.exists(created_path))
        self.assertEqual(created_path, self.hash_file)

    def test_verify_file_hash_success(self):
        """
        Успешная проверка нетронутого файла
        :return: ничего
        """
        save_file_hash(self.test_file)  # Создаем слепок хеша

        is_intact, current, saved = verify_file_hash(self.test_file, self.hash_file)
        self.assertTrue(is_intact)
        self.assertEqual(current, saved)

    def test_verify_file_hash_failure(self):
        """
        Фиксация нарушения целостности
        :return: ничего
        """
        save_file_hash(self.test_file)  # Создаем слепок хеша

        # Портим файл — дописываем байты взлома
        with open(self.test_file, "ab") as f:
            f.write(b"hacked data")

        is_intact, current, saved = verify_file_hash(self.test_file, self.hash_file)
        self.assertFalse(is_intact)  # Целостность должна быть нарушена
        self.assertNotEqual(current, saved)


if __name__ == "__main__":
    unittest.main()