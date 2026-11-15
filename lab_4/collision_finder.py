"""
Модуль для поиска коллизий в укороченных хешах SHA-256.

Реализует атаку "дней рождения" (birthday attack) для поиска двух различных
сообщений, укороченные хеши которых совпадают. Поддерживает длины хеша:
8, 12 и 16 бит.
"""
import hashlib
import random
import string
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union
from tqdm import tqdm

@dataclass(frozen=True)
class CollisionResult:
    """Результат поиска коллизии."""
    msg1: Optional[bytes]
    msg2: Optional[bytes]
    short_hash: Optional[str]
    attempts: int

    @property
    def found(self) -> bool:
        return self.msg1 is not None and self.msg2 is not None



class ShortHashCollisionFinder:
    """
    Класс для поиска коллизий в укороченных хешах SHA-256.

    Поддерживает усечение хешей до 8, 12 или 16 бит. Использует метод
    "дней рождения" для эффективного поиска коллизий.

    Атрибуты:
        VALID_BITS (set): Множество допустимых длин хеша (8, 12, 16).
        trunc_bits (int): Количество бит после усечения.
        trunc_bytes (int): Количество байт для усечения (1 для 8 бит, 2 для 12/16).
        bit_mask (int): Маска для обрезки бит до нужной длины.

    """

    VALID_BITS = {8, 12, 16}

    def __init__(self, trunc_bits: int = 8):
        """
        Инициализация объекта для поиска коллизий.

        Аргументы:
            trunc_bits (int): Количество бит для усечения хеша (8, 12 или 16).
                              По умолчанию 8.

        Возвращает:
            None

        Исключения:
            ValueError: Если trunc_bits не входит в {8, 12, 16}.
        """
        if trunc_bits not in self.VALID_BITS:
            raise ValueError(f"Некорректное количество бит: {trunc_bits}. Допустимо: 8, 12, 16")

        self.trunc_bits = trunc_bits

        # Для 8 бит - 1 байт, для 12 и 16 бит - 2 байта
        if trunc_bits <= 8:
            self.trunc_bytes = 1
        else:
            self.trunc_bytes = 2

        # Маска для обрезки бит
        self.bit_mask = (1 << trunc_bits) - 1

    def _compute_hash(self, data: bytes) -> int:
        """
        Вычисляет укороченный хеш SHA-256 от переданных данных.

        Алгоритм:
            1. Вычисляется полный SHA-256 хеш
            2. Берутся первые trunc_bytes байт
            3. Преобразуются в целое число (big-endian)
            4. Обрезаются до trunc_bits бит с помощью маски

        Аргументы:
            data (bytes): Входные данные в байтовом представлении.

        Возвращает:
            int: Укороченный хеш в диапазоне [0, 2^{trunc_bits} - 1].
        """
        # Полный SHA-256 хеш
        
        full_hash = hashlib.sha256(data).digest()

        # Берём нужное количество байт
        truncated_bytes = full_hash[:self.trunc_bytes]

        # Преобразуем в число (big-endian)
        hash_int = int.from_bytes(truncated_bytes, byteorder='big')

        # Обрезаем до нужного количества бит
        return hash_int & self.bit_mask

    def short_hash_hex(self, data: bytes) -> str:
        """
        Возвращает укороченный хеш в HEX-формате.

        Аргументы:
            data (bytes): Входные данные в байтовом представлении.

        Возвращает:
            str: Шестнадцатеричное представление укороченного хеша
                 (без префикса '0x').
        """
        bits = self._compute_hash(data)
        hex_len = (self.trunc_bits + 3) // 4
        return format(bits, f'0{hex_len}x')

    def _random_string(self) -> bytes:
        """
        Генерирует случайную строку для использования в поиске коллизий.

        Длина строки выбирается случайно от 8 до 16 символов.
        Символы: латинские буквы (A-Z, a-z) и цифры (0-9).

        Возвращает:
            bytes: Случайная строка в байтовом представлении.
        """
        length = random.randint(8, 16)
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for _ in range(length)).encode()

    def find_collision(
        self,
        max_attempts: int = 100000,
        show_progress: bool = True
    ) -> CollisionResult:
        """
        Ищет коллизию — два разных сообщения с одинаковым укороченным хешем.

        Реализует атаку "дней рождения":
            1. Создаётся словарь для хранения ранее встреченных хешей
            2. Для каждой попытки генерируется случайное сообщение
            3. Вычисляется его укороченный хеш
            4. Если хеш уже встречался и сообщения разные → коллизия найдена
            5. Иначе хеш и сообщение сохраняются в словарь

        Аргументы:
            max_attempts (int): Максимальное количество попыток. По умолчанию 100000.
            show_progress (bool): Показывать ли прогресс-бар. По умолчанию True.

        Возвращает:
            Tuple[Optional[bytes], Optional[bytes], Optional[str], int]:
                - msg1 (bytes): Первое сообщение (или None, если коллизия не найдена)
                - msg2 (bytes): Второе сообщение (или None)
                - short_hash (str): HEX-представление общего хеша (или None)
                - attempts (int): Фактическое количество попыток
        """
        seen = {}

        iterator = range(max_attempts)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=f"Поиск коллизии ({self.trunc_bits} бит)",
                unit="попытка"
            )

        for attempt in iterator:
            candidate = self._random_string()
            hash_val = self._compute_hash(candidate)

            if hash_val in seen:
                existing = seen[hash_val]
                if existing != candidate:
                    # Дополнительная проверка перед возвратом
                    if self._compute_hash(existing) == self._compute_hash(candidate):
                        return CollisionResult(
                            msg1=existing,
                            msg2=candidate,
                            short_hash=self.short_hash_hex(candidate),
                            attempts=attempt + 1,
                        )
            else:
                seen[hash_val] = candidate

        return CollisionResult(msg1=None, msg2=None, short_hash=None, attempts=max_attempts)    

    def verify_collision(self, msg1: bytes, msg2: bytes) -> Tuple[bool, int, int]:
        """
        Проверяет, образуют ли два сообщения коллизию.

        Вычисляет укороченные хеши для обоих сообщений и сравнивает их.

        Аргументы:
            msg1 (bytes): Первое сообщение.
            msg2 (bytes): Второе сообщение.

        Возвращает:
            CollisionResult: результат с полями msg1, msg2, short_hash, attempts.
            Если коллизия не найдена, msg1/msg2/short_hash = None.
        """
        hash1 = self._compute_hash(msg1)
        hash2 = self._compute_hash(msg2)
        return (hash1 == hash2, hash1, hash2)

    def theoretical_expected_attempts(self) -> int:
        """
        Вычисляет теоретическое ожидаемое количество попыток для нахождения коллизии.

        Используется формула для атаки "дней рождения":
            E ≈ √(π * n / 2)
        где n = 2^{trunc_bits} — размер пространства хешей.

        Возвращает:
            int: Ожидаемое количество попыток (округляется вниз).

        """
        n = 2 ** self.trunc_bits
        return int(math.sqrt(math.pi * n / 2))

    def theoretical_probability(self, attempts: int) -> float:
        """
        Вычисляет теоретическую вероятность найти коллизию за указанное число попыток.

        Используется формула:
            P ≈ 1 - exp(-attempts * (attempts - 1) / (2 * n))
        где n = 2^{trunc_bits} — размер пространства хешей.

        Аргументы:
            attempts (int): Количество попыток.

        Возвращает:
            float: Вероятность найти коллизию (от 0 до 1).
        """
        if attempts <= 1:
            return 0.0
        n = 2 ** self.trunc_bits
        return 1 - math.exp(-attempts * (attempts - 1) / (2 * n))

    def run_experiments(
        self,
        num_experiments: int = 10,
        max_attempts: int = 50000
    ) -> Dict[int, Optional[int]]:
        """
        Проводит серию экспериментов по поиску коллизий.

        Каждый эксперимент запускает поиск коллизии с заданными параметрами
        и записывает количество попыток или None, если коллизия не найдена.

        Аргументы:
            num_experiments (int): Количество экспериментов. По умолчанию 10.
            max_attempts (int): Максимальное количество попыток на эксперимент.
                                По умолчанию 50000.

        Возвращает:
            Dict[int, Optional[int]]: Словарь, где ключ — номер эксперимента,
                                     значение — количество попыток или None.
        """
        results = {}
        print(f"\nПроведение {num_experiments} экспериментов для {self.trunc_bits} бит...")

        for i in range(num_experiments):
            result = self.find_collision(max_attempts=max_attempts, show_progress=False)

            if result.found:
                is_valid, _, _ = self.verify_collision(result.msg1, result.msg2)
                if is_valid:
                    results[i + 1] = result.attempts
                    print(f"  Эксперимент {i+1}: коллизия найдена за {result.attempts} попыток ✓")
                else:
                    results[i + 1] = None
                    print(f"  Эксперимент {i+1}: ложная коллизия ✗")
            else:
                results[i + 1] = None
                print(f"  Эксперимент {i+1}: коллизия не найдена за {max_attempts} попыток")

        return results

    def get_stats(self, experiments_results: Dict[int, Optional[int]]) -> Dict[str, Union[float, int, str]]:
        """
        Вычисляет статистику по результатам экспериментов.

        Анализирует успешные эксперименты и вычисляет среднее количество попыток,
        минимальное, максимальное, а также долю успешных экспериментов.

        Аргументы:
            experiments_results (Dict[int, Optional[int]]): Результаты экспериментов,
                где ключ — номер эксперимента, значение — количество попыток или None.

        Возвращает:
            Dict[str, Union[float, int, str]]: Словарь со статистикой:
                - 'successful' (int): Количество успешных экспериментов
                - 'total' (int): Общее количество экспериментов
                - 'success_rate' (float): Доля успешных экспериментов (0-1)
                - 'theoretical_expected' (int): Теоретическое ожидание
                - 'average_attempts' (float): Среднее количество попыток
                - 'min_attempts' (int): Минимальное количество попыток
                - 'max_attempts' (int): Максимальное количество попыток
        """
        successful = [v for v in experiments_results.values() if v is not None]
        total = len(experiments_results)

        stats = {
            'successful': len(successful),
            'total': total,
            'success_rate': len(successful) / total if total > 0 else 0.0,
            'theoretical_expected': self.theoretical_expected_attempts()
        }

        if successful:
            stats['average_attempts'] = sum(successful) / len(successful)
            stats['min_attempts'] = min(successful)
            stats['max_attempts'] = max(successful)
        else:
            stats['average_attempts'] = 0.0
            stats['min_attempts'] = None
            stats['max_attempts'] = None

        return stats