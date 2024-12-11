import logging
import math
import mpmath

from workfiles import *
from log import logger

PI = {1: 0.2148, 2: 0.3672, 3: 0.2305, 4: 0.1875}


def bitwise_test(sequence: str) -> float:
    """frequency bit test

    Args:
        sequence (str): pseudorandom sequence of 1 and 0

    Returns:
        float: P-value
    """
    try:
        length = len(sequence)
        if length > 0:
            sum = 0
            for bit in sequence:
                if int(bit):
                    sum -= 1
                else:
                    sum += 1
            s = math.fabs(sum) / (length**0.5)
            logger.info("Complete bitwise test")
            return math.erfc(s / (2**0.5))
    except Exception as error:
        logging.error(error)


def same_bits_test(sequence) -> float:
    """
    func that show how many one-typed bits are there
    Args:
        sequence: a given sequence

    Returns:
        how many one-typed bits are there

    """

    try:
        if len(sequence) == 0: return None
        counter = sequence.count("1")
        counter *= 1 / len(sequence)
        if abs(counter - 0.5) < 2 / math.sqrt(len(sequence)):
            v = 0
            for i in range(len(sequence) - 1):
                if sequence[i] != sequence[i + 1]:
                    v += 1
            num = abs(v - 2 * len(sequence) * counter * (1 - counter))
            denom = 2 * math.sqrt(2 * len(sequence)) * counter * (1 - counter)
            p_value = math.erfc(num / denom)
        else:
            p_value = 0
        logger.info("Complete bitwise test")
        return p_value
    except Exception as ex:
        logging.error(f"ZeroDivisionError: {ex.message}\n{ex.args}\n")


def split_bits(sequence) -> list:
    """
    func than splits sequence in blocks of 8 bits
    Args:
        sequence: given sequence

    Returns:
        blocks formed from sequence on 8

    """

    blocks = []
    quantity = len(sequence) - (len(sequence) % 8)
    for i in range(0, quantity, 8):
        block = sequence[i: i + 8]
        blocks.append(block)
    logger.info("Bits splitted")
    return blocks


def largest_number_of_units(blocks: list) -> dict:
    """
    func that check how many one-typed units are in blocks
    Args:
        blocks: made up blocks

    Returns:
        a sorted dictionary

    """

    try:
        unit_counts = {}
        for block in blocks:
            counter = 0
            max_counter = 0
            for i in block:
                if int(i) == 1:
                    counter += 1
                    max_counter = max(max_counter, counter)
                    if max_counter > 4:
                        max_counter = 4
                else:
                    counter = 0
            if max_counter in unit_counts:
                unit_counts[max_counter] += 1
            else:
                unit_counts[max_counter] = 1
        sorted_dict = dict(sorted(unit_counts.items(), key=lambda x: x[1]))
        logger.info("Dictionary sorted")
        return sorted_dict
    except Exception as ex:
        logging.error(f"TypeError block wasn't str: {ex.message}\n{ex.args}\n")


def length_test(dictionary: dict) -> float:
    """
    func that shows the longest sequence in block
    Args:
        dictionary: given dict

    Returns:
        number of longest sequence

    """

    try:
        if largest_number_of_units(split_bits(dictionary)) == 0: return 0
        square_x = 0
        for i, value in dictionary.items():
            square_x += pow(value - 16 * PI[i], 2) / (16 * PI[i])
        p_value = mpmath.gammainc(3 / 2, square_x / 2)
        logger.info("Longest sequence found")
        return p_value
    except Exception as ex:
        logging.error(
            f"Length of the dictionary is longer than number of pi-constants: {ex.message}\n{ex.args}\n"
        )


if __name__ == "__main__":
    sequence_java = read_text('sequence_java.txt')
    print(largest_number_of_units(split_bits('')))
    print(bitwise_test(sequence_java))
    print(same_bits_test(sequence_java))
    print(
        length_test(
            largest_number_of_units(split_bits(sequence_java))
        )
    )
    sequence_cpp = read_text('sequence_cpp.txt')
    print(bitwise_test(sequence_cpp))
    print(same_bits_test(sequence_cpp))
    print(
        length_test(
            largest_number_of_units(split_bits(sequence_cpp))
        )
    )
