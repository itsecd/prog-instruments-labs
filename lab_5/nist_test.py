import math
from scipy.special import gammaincc
import logging

logger = logging.getLogger(__name__)


def frequency_test(bits):
    logger.info("Запуск частотного теста")
    logger.debug("Длина последовательности: %d", len(bits))

    if not bits:
        raise ValueError("Последовательность пустая")

    sum_bit = 0
    zero_count = 0
    one_count = 0

    for b in bits:
        if b == "0":
            sum_bit -= 1
            zero_count += 1
        elif b == "1":
            sum_bit += 1
            one_count += 1
        else:
            logger.warning("Невалидный символ: '%s'", b)

    logger.debug("Нули: %d, Единицы: %d", zero_count, one_count)

    s = abs(sum_bit) / math.sqrt(len(bits))
    p_value = math.erfc(s / math.sqrt(2))

    logger.info("Частотный тест завершён. p-value = %s", p_value)
    return p_value


def runs_test(bits):
    logger.info("Запуск теста на серии")
    logger.debug("Длина последовательности: %d", len(bits))

    if not bits:
        raise ValueError("Пустая последовательность")

    share_ones = sum(1 for b in bits if b == "1") / len(bits)

    tolerance = 2 / math.sqrt(len(bits))
    if abs(share_ones - 0.5) >= tolerance:
        logger.warning("Доля единиц за пределами нормы: %.4f", share_ones)
        return 0.0

    series = sum(1 for i in range(len(bits) - 1) if bits[i] != bits[i+1])

    numerator = abs(series - 2 * len(bits) * share_ones * (1 - share_ones))
    denominator = 2 * math.sqrt(2 * len(bits)) * share_ones * (1 - share_ones)

    p_value = math.erfc(numerator / denominator)

    logger.info("Тест на серии завершён. p-value = %s", p_value)
    return p_value


def longest_run_test(bits, block_size=8):
    logger.info("Запуск теста на длинные серии (block=%d)", block_size)

    n = len(bits)
    if n % block_size != 0:
        raise ValueError("Длина должна быть кратна размеру блока")

    num_blocks = n // block_size

    pi = [0.2148, 0.3672, 0.2305, 0.1875]
    v = [0, 0, 0, 0]

    for i in range(num_blocks):
        block = bits[i*block_size:(i+1)*block_size]
        max_run = 0
        cur = 0

        for b in block:
            if b == "1":
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 0

        if max_run <= 1:
            v[0] += 1
        elif max_run == 2:
            v[1] += 1
        elif max_run == 3:
            v[2] += 1
        else:
            v[3] += 1

    chi = 0
    for i in range(4):
        exp = pi[i] * num_blocks
        chi += (v[i] - exp) ** 2 / exp

    p_value = gammaincc(1.5, chi / 2)

    logger.info("Тест на длинные серии завершён. p-value = %s", p_value)
    return p_value
