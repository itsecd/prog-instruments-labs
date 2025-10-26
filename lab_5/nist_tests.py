import math
from scipy.special import gammaincc
import logging


def frequency_monobit_test(binary_sequence):
    """
    Частотный побитовый тест NIST
    :param binary_sequence: Последовательность(строка) из "0" и "1"
    :return: P-значение последовательности
    """
    logging.debug("Начало выполнения frequency_monobit_test")

    n = len(binary_sequence)

    sum_sn= 0

    for bit in binary_sequence:
        sum_sn += 1 if bit == '1' else -1
    logging.debug(f"Сумма (+1/-1) для битов: {sum_sn}")

    normal_sum = abs(sum_sn) / math.sqrt(n)
    logging.debug(f"Нормализованная сумма: {normal_sum}")

    p_value = math.erfc(normal_sum / math.sqrt(2))
    logging.debug(f"Рассчитанное p-value: {p_value}")

    return p_value


def runs_test(binary_sequence):
    """
    Тест на одинаковые подряд идущие биты
    :param binary_sequence: Последовательность(строка) из "0" и "1"
    :return: P-значение последовательности
    """
    logging.debug("Начало выполнения runs_test")

    n = len(binary_sequence)
    if n < 2:
        logging.error("Ошибка валидации: слишком короткая последовательность для runs_test (<2 бит)")
        return None

    ones_count = binary_sequence.count('1')
    zeta = ones_count / n
    logging.debug(f"Количество единиц: {ones_count}, доля zeta: {zeta}")

    if abs(zeta - 0.5) >= (2 / math.sqrt(n)):
        logging.error("Ошибка валидации: последовательность не прошла предварительную проверку в runs_test")
        return 0  # Последовательность не прошла проверку

    series = 0
    for i in range(n - 1):
        if binary_sequence[i] != binary_sequence[i + 1]:
            series += 1
    logging.debug(f"Количество серий (runs): {series}")

    numerator = abs(series - 2 * n * zeta * (1 - zeta))
    denominator = 2 * math.sqrt(2 * n) * zeta * (1 - zeta)
    logging.debug(f"Числитель: {numerator}, знаменатель: {denominator}")

    p_value = math.erfc(numerator / denominator)
    logging.debug(f"Рассчитанное p-value: {p_value}")

    return p_value


def longest_run_test(binary_sequence, block_size=8):
    """
    Тест на самую длинную последовательность единиц в блоке
    :param binary_sequence: Последовательность(строка) из "0" и "1"
    :param block_size: Размер блока
    :return: P-значение последовательности
    """
    logging.debug("Начало выполнения longest_run_test")

    n = len(binary_sequence)
    if n % block_size != 0:
        logging.error(f"Ошибка валидации: длина последовательности ({n}) не кратна размеру блока {block_size}")
        return None

    num_blocks = n // block_size
    logging.debug(f"Количество блоков: {num_blocks}")

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

    logging.debug(f"Подсчёт блоков по категориям v: {v}")

    hi_square = sum((v[i] - num_blocks * pi[i]) ** 2 / (num_blocks * pi[i]) for i in range(4))
    logging.debug(f"Вычислено x2: {hi_square}")

    p_value = gammaincc(1.5, hi_square / 2)
    logging.debug(f"Рассчитанное p-value: {p_value}")

    return p_value