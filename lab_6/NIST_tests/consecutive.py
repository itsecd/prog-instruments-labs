import math


def consecutive_bit_test(sequence: str) -> float:
    """
    This function performs a test for identical consecutive bits
    :param sequence: binary sequence
    :return: p-value
    """
    length = len(sequence)
    if length == 0:
        raise ValueError("Sequence is empty")

    one_counter = 0
    for i in sequence:
        if i == '1':
            one_counter += 1
        elif i != '0':
            raise ValueError("Sequence is non binary")

    dzeta = one_counter / length

    if abs(dzeta - 0.5) >= 2 / math.sqrt(length):
        return 0

    v_n = 0
    for i in range(length - 1):
        if sequence[i] != sequence[i + 1]:
            v_n += 1

    denominator = 2 * math.sqrt(2 * length) * dzeta * (1 - dzeta)
    if denominator == 0:
        return 0.0

    p_value = math.erfc(abs(v_n - 2 * length * dzeta * (1 - dzeta)) / denominator)
    return p_value