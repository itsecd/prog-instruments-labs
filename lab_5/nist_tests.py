import math
from scipy.special import gammaincc


def frequency_monobit_test(binary_sequence):
    """
    Частотный побитовый тест NIST
    :param binary_sequence: Последовательность(строка) из "0" и "1"
    :return: P-значение последовательности
    """
    n = len(binary_sequence)

    sum_sn= 0

    for bit in binary_sequence:
        sum_sn += 1 if bit == '1' else -1

    normal_sum = abs(sum_sn) / math.sqrt(n)
    p_value = math.erfc(normal_sum / math.sqrt(2))

    return p_value


def runs_test(binary_sequence):
    """
    Тест на одинаковые подряд идущие биты
    :param binary_sequence: Последовательность(строка) из "0" и "1"
    :return: P-значение последовательности
    """
    n = len(binary_sequence)
    if n < 2:
        raise ValueError("Последовательность должна содержать минимум 2 бита")

    ones_count = binary_sequence.count('1')
    zeta = ones_count / n

    if abs(zeta - 0.5) >= (2 / math.sqrt(n)):
        return 0  # Последовательность не прошла проверку

    series = 0
    for i in range(n - 1):
        if binary_sequence[i] != binary_sequence[i + 1]:
            series += 1

    numerator = abs(series - 2 * n * zeta * (1 - zeta))
    denominator = 2 * math.sqrt(2 * n) * zeta * (1 - zeta)
    p_value = math.erfc(numerator / denominator)

    return p_value


def longest_run_test(binary_sequence, block_size=8):
    """
    Тест на самую длинную последовательность единиц в блоке
    :param binary_sequence: Последовательность(строка) из "0" и "1"
    :param block_size: Размер блока
    :return: P-значение последовательности
    """

    n = len(binary_sequence)
    if n % block_size != 0:
        raise ValueError(f"Длина последовательности ({n}) должна быть кратна размеру блока 8")

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