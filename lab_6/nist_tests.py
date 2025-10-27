import math

from spicy import special


def frequency_bit_test(data: str) -> float:
    """
    The function checks if sequence is random
    using frequency bit test
    :param data: sequence
    :return: result
    """
    s_n = (data.count("1") - data.count("0")) / math.sqrt(len(data))
    p_value = math.erfc(abs(s_n) / math.sqrt(2))
    return p_value


def equally_consecutive_bits(data: str) -> float:
    """
    The function checks if sequence is random
    using a test for identical consecutive bits
    :param data: sequence
    :return: result
    """
    seq_len = len(data)
    zeta = data.count("1") / seq_len

    if not (abs(zeta - 0.5) < (2 / math.sqrt(seq_len))):
        return 0

    v_n = 0
    for i in range(seq_len - 1):
        if data[i] != data[i + 1]:
            v_n += 1

    numerator = abs(v_n - 2 * seq_len * zeta * (1 - zeta))
    denominator = 2 * math.sqrt(2 * seq_len) * zeta * (1 - zeta)
    p_value = math.erfc(abs(numerator) / denominator)
    return p_value


def longest_sequence_test(data: str) -> float:
    """
    The function checks if sequence is random
    using test for the longest sequence in block
    :param data: sequence
    :return: result
    """
    blocks = [data[i : i + 8] for i in range(0, len(data), 8)]
    v = [0, 0, 0, 0]

    for block in blocks:
        mx_length = float("-inf")
        curr_length = 0
        for digit in block:
            if digit == "1":
                curr_length += 1
                mx_length = max(mx_length, curr_length)
            else:
                curr_length = 0

        match mx_length:
            case mx_length if mx_length <= 1:
                v[0] += 1
            case 2:
                v[1] += 1
            case 3:
                v[2] += 1
            case mx_length if mx_length >= 4:
                v[3] += 1

    p = [0.2148, 0.3672, 0.2305, 0.1875]
    xi_square = sum(((v[i] - 16 * p[i]) ** 2) / (16 * p[i]) for i in range(len(v)))
    p_value = special.gammainc((3 / 2), (xi_square / 2))
    return p_value