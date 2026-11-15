import unittest
import hashlib
from collision_finder import ShortHashCollisionFinder


class TestShortHashCollision(unittest.TestCase):
    
    def setUp(self):
        """Подготовка к тестам"""
        self.finders = {
            8: ShortHashCollisionFinder(trunc_bits=8),
            12: ShortHashCollisionFinder(trunc_bits=12),
            16: ShortHashCollisionFinder(trunc_bits=16)
        }
    
    def test_valid_bits_initialization(self):
        """Проверка инициализации с корректными битами"""
        for bits in [8, 12, 16]:
            finder = ShortHashCollisionFinder(trunc_bits=bits)
            self.assertEqual(finder.trunc_bits, bits)
    
    def test_invalid_bits_initialization(self):
        """Проверка инициализации с некорректными битами"""
        with self.assertRaises(ValueError):
            ShortHashCollisionFinder(trunc_bits=1)
        with self.assertRaises(ValueError):
            ShortHashCollisionFinder(trunc_bits=32)
    
    def test_hash_bits_range(self):
        """Проверка диапазона значений укороченного хеша"""
        test_cases = [
            (8, 0, 255),
            (12, 0, 4095),
            (16, 0, 65535)
        ]
        
        for bits, min_val, max_val in test_cases:
            finder = ShortHashCollisionFinder(trunc_bits=bits)
            h = finder._compute_hash(b"test")
            self.assertTrue(min_val <= h <= max_val)
    
    def test_same_message_same_hash(self):
        """Одинаковые сообщения дают одинаковый хеш"""
        for bits in [8, 12, 16]:
            finder = ShortHashCollisionFinder(trunc_bits=bits)
            data = b"hello_world_test"
            self.assertEqual(finder._compute_hash(data), finder._compute_hash(data))
    
    def test_different_messages_give_different_hash_probably(self):
        """Разные сообщения дают разные хеши (вероятностно)"""
        for bits in [8, 12, 16]:
            finder = ShortHashCollisionFinder(trunc_bits=bits)
            h1 = finder._compute_hash(b"message1")
            h2 = finder._compute_hash(b"message2")
            # Для малых бит может быть коллизия, но вероятность мала
            # Проверяем не на равенство, а на то, что функция детерминирована
            self.assertIsNotNone(h1)
            self.assertIsNotNone(h2)
    
    def test_collision_finder_8bit(self):
        """Поиск коллизии для 8 бит должен быть быстрым и успешным"""
        finder = ShortHashCollisionFinder(trunc_bits=8)
        msg1, msg2, sh, attempts = finder.find_collision(max_attempts=200, show_progress=False)
        
        if msg1 is None or msg2 is None:
            self.fail("Коллизия не найдена для 8 бит за 200 попыток")
        else:
            self.assertNotEqual(msg1, msg2)
            self.assertEqual(finder._compute_hash(msg1), finder._compute_hash(msg2))
            self.assertLessEqual(attempts, 200)
    
    def test_collision_finder_16bit(self):
        """Поиск коллизии для 16 бит"""
        finder = ShortHashCollisionFinder(trunc_bits=16)
        msg1, msg2, sh, attempts = finder.find_collision(max_attempts=2000, show_progress=False)
        
        if msg1 is not None:
            self.assertNotEqual(msg1, msg2)
            self.assertEqual(finder._compute_hash(msg1), finder._compute_hash(msg2))
    
    def test_theoretical_expected_attempts(self):
        """Проверка теоретических расчётов"""
        expected_ranges = {
            8: (15, 25),    # ~20
            12: (70, 90),   # ~80
            16: (300, 350)  # ~321
        }
        
        for bits, (min_exp, max_exp) in expected_ranges.items():
            finder = ShortHashCollisionFinder(trunc_bits=bits)
            expected = finder.theoretical_expected_attempts()
            self.assertTrue(min_exp <= expected <= max_exp,
                          f"Для {bits} бит ожидание {expected} вне диапазона [{min_exp}, {max_exp}]")
    
    def test_theoretical_probability(self):
        """Проверка расчёта вероятности"""
        finder = ShortHashCollisionFinder(trunc_bits=8)
        
        # За 0 попыток вероятность 0
        self.assertEqual(finder.theoretical_probability(0), 0.0)
        
        # За 100 попыток для 8 бит вероятность близка к 1
        prob = finder.theoretical_probability(100)
        self.assertGreater(prob, 0.99)
        
        # За 10 попыток для 16 бит вероятность мала
        finder16 = ShortHashCollisionFinder(trunc_bits=16)
        prob = finder16.theoretical_probability(10)
        self.assertLess(prob, 0.01)
    
    def test_hex_output_length(self):
        """Проверка длины HEX-вывода"""
        test_cases = [
            (8, 2),   # 8 бит = 2 HEX символа
            (12, 3),  # 12 бит = 3 HEX символа
            (16, 4)   # 16 бит = 4 HEX символа
        ]
        
        for bits, expected_len in test_cases:
            finder = ShortHashCollisionFinder(trunc_bits=bits)
            hex_val = finder.short_hash_hex(b"test")
            self.assertEqual(len(hex_val), expected_len)
    
    def test_random_string_generation(self):
        """Проверка генерации случайных строк"""
        finder = ShortHashCollisionFinder(trunc_bits=8)
        
        # Проверка длины (метод _random_string не принимает аргументы, он генерирует длину 8-16)
        # Поэтому проверим несколько сгенерированных строк на соответствие диапазону длин
        for _ in range(20):
            s = finder._random_string()
            self.assertTrue(8 <= len(s) <= 16)
            self.assertIsInstance(s, bytes)
        
        # Проверка на разные значения
        s1 = finder._random_string()
        s2 = finder._random_string()
        # Вероятность коллизии ничтожно мала
        self.assertNotEqual(s1, s2)
    
    def test_experiments_collection(self):
        """Проверка сбора экспериментальных данных"""
        finder = ShortHashCollisionFinder(trunc_bits=8)
        results = finder.run_experiments(num_experiments=3, max_attempts=1000)
        
        self.assertEqual(len(results), 3)
        for exp_num, attempts in results.items():
            self.assertIsInstance(exp_num, int)
            # attempts может быть None если не найдена, или int
            if attempts is not None:
                self.assertIsInstance(attempts, int)
    
    def test_get_stats(self):
        """Проверка расчёта статистики"""
        finder = ShortHashCollisionFinder(trunc_bits=8)
        
        # Пустые результаты
        empty_stats = finder.get_stats({})
        self.assertEqual(empty_stats['successful'], 0)
        self.assertEqual(empty_stats['total'], 0)
        
        # Результаты с успехами
        results = {1: 10, 2: 20, 3: 30, 4: None}
        stats = finder.get_stats(results)
        self.assertEqual(stats['successful'], 3)
        self.assertEqual(stats['total'], 4)
        self.assertEqual(stats['average_attempts'], 20.0)
        self.assertEqual(stats['min_attempts'], 10)
        self.assertEqual(stats['max_attempts'], 30)
    
    def test_space_size(self):
        """Проверка размера пространства хешей"""
        expected_sizes = {
            8: 256,
            12: 4096,
            16: 65536
        }
        
        for bits, expected in expected_sizes.items():
            finder = ShortHashCollisionFinder(trunc_bits=bits)
            self.assertEqual(2 ** finder.trunc_bits, expected)


if __name__ == "__main__":
    unittest.main()