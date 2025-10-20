from scipy.special import gammaincc as igamc


def frequency_bitwise_test(sequence: str) -> float:
    """
    The function performs a frequency bitwise test for a given sequence

    :param sequence: Binary string
    :return: P-value
    """
    s_n = 0
    n = len(sequence)
    for val in sequence:
        s_n += 1 / math.sqrt(n) if val == "1" else -1 / math.sqrt(n)

    p_value = math.erfc(abs(s_n) / math.sqrt(2))
    return p_value


def identical_consecutive_bits(sequence: str) -> float:
    """
    The function performs a test for the same consecutive bits for a given string
    :param sequence: Binary string
    :return: P-value
    """
    n = len(sequence)
    s_n = sum(1 for val in sequence if val == "1") / n

    if abs(s_n - 0.5) >= 2 / math.sqrt(n):
        return 0.0

    v_n = sum(1 for i in range(n - 1) if sequence[i] != sequence[i + 1])
    p_value = math.erfc(
        (abs(v_n - 2 * n * s_n * (1 - s_n))) / (2 * math.sqrt(2 * n) * s_n * (1 - s_n))
    )

    return p_value


import math


def longest_tes_sequence(sequence: str, pi: list[float], block_size: int) -> float:
    """
    The function checks the longest sequence of elements in a block for a given sequence.
    :param sequence: Binary string
    :param pi: Probabilities [p1, p2, p3, p4]
    :param block_size: Block length
    :return: P-value
    """

    n = len(sequence)
    v = [0, 0, 0, 0]
    blocks = [sequence[i : i + block_size] for i in range(0, n, block_size)]

    for block in blocks:
        max_block = 0
        max_val = 0
        for val in block:
            if val == "1":
                max_val += 1
                max_block = max(max_val, max_block)
            else:
                max_val = 0
        match max_block:
            case _ if max_block <= 1:
                v[0] += 1
            case 2:
                v[1] += 1
            case 3:
                v[2] += 1
            case _ if max_block >= 4:
                v[3] += 1

    chi_square = sum(((v[i] - 16 * pi[i]) ** 2) / (16 * pi[i]) for i in range(4))
    p_value = igamc(3 / 2, chi_square / 2)
    return p_value
