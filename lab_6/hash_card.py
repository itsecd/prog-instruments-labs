import hashlib
import json
import os
import time
import multiprocessing
from typing import Optional, List, Tuple
from multiprocessing import Event, Queue, Value, Lock
from const import *


class CardSearcher:
	"""Класс для поиска номера кредитной карты по его BLAKE2s хешу.

	Используется многопроцессорная обработка для ускорения поиска.
	"""

	def __init__(self):
		"""Инициализирует CardSearcher с примитивами синхронизации.

		Создает:
			- found_event: Event для остановки процессов при нахождении результата
			- result_queue: Queue для передачи найденного номера карты
			- processed_count: Value(int) счетчик обработанных номеров
			- lock: Lock для синхронизации доступа к счетчику
		"""
		self.found_event = Event()
		self.result_queue = Queue()
		self.processed_count = Value('i', 0)
		self.lock = Lock()

	def get_cpu_count(self) -> int:
		"""Возвращает количество доступных CPU ядер.

		Returns:
			int: Количество CPU ядер (минимум 1)
		"""
		try:
			return os.cpu_count() or 1
		except:
			return 1

	def generate_card_numbers(self, bin_code: str) -> List[str]:
		"""Генерирует список возможных номеров карт для данного BIN.

		Args:
			bin_code: 6-значный BIN код карты

		Returns:
			List[str]: Список из 1,000,000 номеров карт в формате BIN + 6 цифр + LAST_4_DIGITS
		"""
		return [f"{bin_code}{i:06}{LAST_4_DIGITS}" for i in range(1000000)]

	def search_card_number(self, num_processes=None) -> Tuple[Optional[str], float]:
		"""Основной метод поиска номера карты.

		Запускает многопроцессорный поиск по всем BIN-кодам.

		Args:
			num_processes: Число процессов (по умолчанию равняется числу доступных ядер)

		Returns:
			Tuple[Optional[str], float]: Найденный номер карты и время выполнения
		"""
		cpu_count = num_processes or self.get_cpu_count()
		start_time = time.time()

		with multiprocessing.Pool(
				cpu_count,
				initializer=self.init_worker,
				initargs=(self.found_event, self.result_queue, self.processed_count, self.lock)
		) as pool:
			for bin_code in BINS:
				if self.found_event.is_set():
					break

				card_numbers = self.generate_card_numbers(bin_code)
				for _ in pool.imap_unordered(
						self.check_hash_wrapper,
						card_numbers,
						chunksize=1000
				):
					if self.found_event.is_set():
						break

		end_time = time.time()
		elapsed_time = end_time - start_time
		result = self.result_queue.get() if not self.result_queue.empty() else None
		return result, elapsed_time

	@staticmethod
	def init_worker(found_event, result_queue, processed_count, lock):
		"""Инициализирует глобальные переменные в каждом рабочем процессе.

		Args:
			found_event: Event для сигнала остановки
			result_queue: Queue для передачи результата
			processed_count: Value(int) счетчик обработанных номеров
			lock: Lock для синхронизации
		"""
		global worker_event, worker_queue, worker_count, worker_lock
		worker_event = found_event
		worker_queue = result_queue
		worker_count = processed_count
		worker_lock = lock

	@staticmethod
	def check_hash_wrapper(card_number: str) -> Optional[str]:
		"""Проверяет хеш номера карты на соответствие целевому.

		Args:
			card_number: Номер карты для проверки

		Returns:
			Optional[str]: Номер карты если хеш совпал, иначе None
		"""
		if worker_event.is_set():
			return None

		with worker_lock:
			worker_count.value += 1

		if hashlib.blake2s(card_number.encode()).hexdigest() == CARD_HASH:
			worker_event.set()
			worker_queue.put(card_number)
			return card_number
		return None

	@staticmethod
	def save_result(card_number: str, path: str = "result.json") -> None:
		"""Сохраняет найденный номер карты в JSON файл.

		Args:
			card_number: Найденный номер карты
			path: Путь для сохранения файла (по умолчанию "result.json")
		"""
		data = {"card_number": card_number}
		with open(path, 'w', encoding='utf-8') as f:
			json.dump(data, f, indent=2)