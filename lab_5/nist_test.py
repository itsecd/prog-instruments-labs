import math
from msilib import sequence

from scipy.special import gammaincc


def frequency_test(bits):
	"""
	Частотный тест NIST.

	Проверяет частоту появления нулей и единиц в последовательности.

	Параметры:
	bits (str): строка, содержащая последовательность нулей и единиц.

	Возвращает:
	float: p-значение теста1.
	"""

	sum_bit = 0
	for bit in bits:
		if bit == "0":
			sum_bit += -1
		else:
			sum_bit += 1

	# Вычисляем абсолютное значение нормализованной суммы
	abs_normal_sum = abs(sum_bit) / math.sqrt(len(bits))

	# Вычисляем p-value с использованием дополнительной функции ошибок
	p_value = math.erfc(abs_normal_sum / math.sqrt(2))
	return p_value


def runs_test(bits):
	"""
	Тест на одинаковые подряд идущие биты.

	Проверяет частоту смены знаков в последовательности.

	Параметры:
	bits (str): строка, содержащая последовательность нулей и единиц.

	Возвращает:
	float: p-значение теста.
	"""
	share_units = 0
	for bit in bits:
		if bit == "1":
			share_units += 1
	share_units = share_units / len(bits)

	if abs(share_units - 0.5) >= 2 / math.sqrt(len(bits)):
		return 0.0

	series = 0
	for i in range(len(bits) - 1):
		if bits[i] != bits[i + 1]:
			series += 1

	numerator = abs(series - 2 * len(bits) * share_units * (1 - share_units))
	denominator = 2 * math.sqrt(2 * len(bits)) * share_units * (1 - share_units)
	p_value = math.erfc(numerator / denominator)
	return p_value


def longest_run_test(binary_sequence, block_size=8):
	"""
	Тест на самую длинную последовательность единиц в блоке.

	Проверяет длину самой длинной последовательности единиц в каждом блоке.

	Параметры:
	binary_sequence (str): строка, содержащая последовательность нулей и единиц.
	block_size (int): размер блока (по умолчанию 8).

	Возвращает:
	float: p-значение теста.
	"""
	n = len(binary_sequence)
	if n % block_size != 0:
		raise ValueError(f"Длина последовательности ({n}) должна быть кратна размеру блока ({block_size})")

	num_blocks = n // block_size

	pi = [0.2148, 0.3672, 0.2305, 0.1875]

	v = [0, 0, 0, 0]  # v0, v1, v2, v3

	for i in range(num_blocks):
		block = binary_sequence[i * block_size: (i + 1) * block_size]
		max_run = 0
		current_run = 0

		for bit in block:
			if bit == '1':
				current_run += 1
				max_run = max(max_run, current_run)
			else:
				current_run = 0

		if max_run <= 1:
			v[0] += 1
		elif max_run == 2:
			v[1] += 1
		elif max_run == 3:
			v[2] += 1
		else:  # max_run >= 4
			v[3] += 1

	hi_square = sum((v[i] - num_blocks * pi[i]) ** 2 / (num_blocks * pi[i]) for i in range(4))

	p_value = gammaincc(1.5, hi_square / 2)

	return p_value
