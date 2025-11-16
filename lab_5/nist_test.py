python
import math
from msilib import sequence
from scipy.special import gammaincc
import logging

logger = logging.getLogger(__name__)


def frequency_test(bits):
    """
    Частотный тест NIST.

    Проверяет частоту появления нулей и единиц в последовательности.

    Параметры:
    bits (str): строка, содержащая последовательность нулей и единиц.

    Возвращает:
    float: p-значение теста.
    """
    logger.info("Запуск частотного теста NIST")
    logger.debug(f"Длина входной последовательности: {len(bits)} бит")

    if not bits:
        logger.error("Получена пустая последовательность для частотного теста")
        raise ValueError("Последовательность не может быть пустой")

    sum_bit = 0
    zero_count = 0
    one_count = 0

    for bit in bits:
        if bit == "0":
            sum_bit += -1
            zero_count += 1
        elif bit == "1":
            sum_bit += 1
            one_count += 1
        else:
            logger.warning(f"Обнаружен невалидный символ в последовательности: '{bit}'")

    logger.debug(f"Статистика частотного теста: нули={zero_count}, единицы={one_count}")
    logger.debug(f"Сумма битов (нормализованная): {sum_bit}")

    abs_normal_sum = abs(sum_bit) / math.sqrt(len(bits))
    logger.debug(f"Абсолютное нормализованное значение: {abs_normal_sum}")

    p_value = math.erfc(abs_normal_sum / math.sqrt(2))

    logger.info(f"Частотный тест завершен. p-значение: {p_value}")

    if p_value < 0.01:
        logger.warning(f"Низкое p-значение в частотном тесте: {p_value}")
    elif p_value > 0.99:
        logger.warning(f"Высокое p-значение в частотном тесте: {p_value}")

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
    logger.info("Запуск теста на серии (runs test)")
    logger.debug(f"Длина входной последовательности: {len(bits)} бит")

    if not bits:
        logger.error("Получена пустая последовательность для теста на серии")
        raise ValueError("Последовательность не может быть пустой")

    valid_bits = [b for b in bits if b in '01']
    if len(valid_bits) != len(bits):
        invalid_count = len(bits) - len(valid_bits)
        logger.warning(f"Обнаружено {invalid_count} невалидных символов в последовательности")

    share_units = 0
    for bit in bits:
        if bit == "1":
            share_units += 1
    share_units = share_units / len(bits)

    logger.debug(f"Доля единиц в последовательности: {share_units:.4f}")

    tolerance = 2 / math.sqrt(len(bits))
    if abs(share_units - 0.5) >= tolerance:
        logger.warning(
            f"Доля единиц ({share_units:.4f}) выходит за допустимые пределы "
            f"[{0.5 - tolerance:.4f}, {0.5 + tolerance:.4f}]. Тест может быть невалидным."
        )
        return 0.0

    series = 0
    for i in range(len(bits) - 1):
        if bits[i] != bits[i + 1]:
            series += 1

    logger.debug(f"Количество серий (смен): {series}")

    numerator = abs(series - 2 * len(bits) * share_units * (1 - share_units))
    denominator = 2 * math.sqrt(2 * len(bits)) * share_units * (1 - share_units)

    logger.debug(f"Числитель: {numerator:.4f}, Знаменатель: {denominator:.4f}")

    p_value = math.erfc(numerator / denominator)

    logger.info(f"Тест на серии завершен. p-значение: {p_value}")

    if p_value < 0.01:
        logger.warning(f"Низкое p-значение в тесте на серии: {p_value}")

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
    logger.info(f"Запуск теста на самую длинную серию (block_size={block_size})")
    logger.debug(f"Длина входной последовательности: {len(binary_sequence)} бит")

    if not binary_sequence:
        logger.error("Получена пустая последовательность для теста на длинные серии")
        raise ValueError("Последовательность не может быть пустой")

    n = len(binary_sequence)
    if n % block_size != 0:
        error_msg = f"Длина последовательности ({n}) должна быть кратна размеру блока ({block_size})"
        logger.error(error_msg)
        raise ValueError(error_msg)

    num_blocks = n // block_size
    logger.debug(f"Количество блоков для анализа: {num_blocks}")

    pi = [0.2148, 0.3672, 0.2305, 0.1875]
    logger.debug(f"Ожидаемые распределения вероятностей: {pi}")

    v = [0, 0, 0, 0]

    logger.info("Начало анализа блоков на максимальные серии единиц...")

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

        if (i + 1) % 1000 == 0:
            logger.debug(f"Проанализировано {i + 1} блоков из {num_blocks}")

    logger.info(f"Анализ блоков завершен. Распределение серий: v={v}")
    logger.debug(f"Ожидаемое распределение: {[pi_i * num_blocks for pi_i in pi]}")

    hi_square = 0
    for i in range(4):
        expected = num_blocks * pi[i]
        observed = v[i]
        component = (observed - expected) ** 2 / expected
        hi_square += component
        logger.debug(f"Категория {i}: наблюдено={observed}, ожидаемо={expected:.2f}, вклад={component:.4f}")

    logger.debug(f"Общая статистика хи-квадрат: {hi_square:.4f}")

    p_value = gammaincc(1.5, hi_square / 2)

    logger.info(f"Тест на самую длинную серию завершен. p-значение: {p_value}")

    if p_value < 0.01:
        logger.warning(f"Низкое p-значение в тесте на длинные серии: {p_value}")
    elif p_value > 0.99:
        logger.warning(f"Высокое p-значение в тесте на длинные серии: {p_value}")

    return p_value
