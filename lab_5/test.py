import math
import mpmath

from works_files import *

PI = {0: 0.2148, 1: 0.3672, 2: 0.2305, 3: 0.1875}
MAX_LENGTH_BLOCK = 8


def tests(mode: Mode, text: str, text_write: str, seq_rand: str) -> None:
    """
    Redirects to the selected test for the sequence.

    Parameters
        mode: a Mode enumeration indicating the mode of operation
        text: the path to the JSON file containing the binary sequence.
        text_write: the path to write the result of the test.
        seq_rand: the key in the dictionary to the binary sequence.
    """
    match mode:
        case Mode.frequency:
            frequency_test(text, text_write, seq_rand)
        case Mode.same_bits:
            same_bits_test(text, text_write, seq_rand)
        case Mode.longest_sequence_in_block:
            longest_sequence_in_block_test(text, text_write, seq_rand)


def frequency_test(text: str, text_write: str, seq_rand: str) -> None:
    """
    Performs a frequency test on a binary sequence and writes the result to a file.

    Parameters
        text: the path to the JSON file containing the binary sequence.
        text_write: the path to write the result of the test.
        seq_rand: the key in the dictionary to the binary sequence.
    """
    seq = read_json(text)

    sequence = [-1 if bit == "0" else 1 for bit in seq.get(seq_rand)]
    s_n = sum(sequence) / math.sqrt(len(sequence))
    p_value = math.erfc(math.fabs(s_n) / math.sqrt(2))
    write_files(text_write, "Частотный побитовый тест " f'{seq_rand} : {p_value}\n')


def same_bits_test(text: str, text_write: str, seq_rand: str) -> None:
    """
    Performs a test for the same consecutive bits and writes the result to a file.

    Parameters
        text: the path to the JSON file containing the binary sequence.
        text_write: the path to write the result of the test.
        seq_rand: the key in the dictionary to the binary sequence.
    """
    seq = read_json(text)

    n = len(seq.get(seq_rand))
    count = seq.get(seq_rand).count("1")
    sum_list = count / n
    if abs(sum_list - 0.5) < (2 / math.sqrt(len(seq.get(seq_rand)))):
        v = 0
        for bit in range(len(seq.get(seq_rand)) - 1):
            if seq.get(seq_rand)[bit] != seq.get(seq_rand)[bit + 1]:
                v += 1
        a_numerator = abs(v - 2 * n * sum_list * (1 - sum_list))
        b_denominator = 2 * math.sqrt(2 * n) * sum_list * (1 - sum_list)
        p_value = math.erfc(a_numerator / b_denominator)
    else:
        p_value = 0
    write_files(text_write, "Тест на одинаковые подряд идущие биты " f'{seq_rand} : {p_value}\n')


def longest_sequence_in_block_test(text: str, text_write: str, seq_rand: str) -> None:
    """
     Performs a test for the longest sequence of ones in the block and writes the result to a file.

    Parameters
        text: the path to the JSON file containing the binary sequence.
        text_write: the path to write the result of the test.
        seq_rand: the key in the dictionary to the binary sequence.
    """
    seq = read_json(text)

    blocks = [seq.get(seq_rand)[i:i + MAX_LENGTH_BLOCK] for i in range(0, len(seq.get(seq_rand)), MAX_LENGTH_BLOCK)]
    v = {1: 0, 2: 0, 3: 0, 4: 0}
    for block in blocks:
        max_length, length = 0, 0
        for bit in block:
            length = length + 1 if bit == "1" else 0
            max_length = max(max_length, length)
        match max_length:
            case 0 | 1:
                v[1] += 1
            case 2:
                v[2] += 1
            case 3:
                v[3] += 1
            case _:
                v[4] += 1
    xi_square = 0
    for i in range(4):
        xi_square += pow(v[i + 1] - 16 * PI[i], 2) / (16 * PI[i])
    value = mpmath.gammainc(3 / 2, xi_square / 2)
    write_files(text_write, "Тест на самую длинную последовательность единиц в блоке " f'{seq_rand} : {value}\n')