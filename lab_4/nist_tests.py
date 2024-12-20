import logging
import math
import mpmath

from work_files import *
from log import logger


PI = {1: 0.2148, 2: 0.3672, 3: 0.2305, 4: 0.1875}


def frequency_bit_test(seq) -> float:
    """The function for frequency bitwise test
    Args:
      seq: generated sequence
    Returns:
      p_val: probability value for the generator
    """
    try:
        summary = 0
        for i in seq:
            if i not in ('0', '1'):
                logging.warning(f"Invalid value in sequence ignored: {i}")
                continue
            if int(i) == 1:
                summary += 1
            else:
                summary -= 1
        summary = math.fabs(summary) / math.sqrt(len(seq))
        p_val = math.erfc(summary / math.sqrt(2))
        logger.info(f"Complete bitwise test, result: {p_val}")
        return p_val
    except Exception as er:
        logging.error(f"ZeroDivisionError: {er.message}\n{er.args}\n")


def ordinary_bit_test(seq) -> float:
    """The function for tests for equal sequential bits
    Args:
      seq: generated sequence
    Returns:
      p_val: probability value for the generator
    """
    try:
        cntr = seq.count("1")
        cntr *= 1 / len(seq)
        if abs(cntr - 0.5) < 2 / math.sqrt(len(seq)):
            v = 0
            for i in range(len(seq) - 1):
                if seq[i] != seq[i + 1]:
                    v += 1
            num = abs(v - 2 * len(seq) * cntr * (1 - cntr))
            denom = 2 * math.sqrt(2 * len(seq)) * cntr * (1 - cntr)
            p_val = math.erfc(num / denom)
        else:
            p_val = 0
        logger.info(f"Complete bitwise test successfully, result: {p_val}")
        return p_val
    except Exception as er:
        logging.error(f"ZeroDivisionError: {er.message}\n{er.args}\n")


def split_bits(seq) -> list:
    """The function for separating a sequence into blocks
    Args:
      seq: generated sequence
    Returns:
      blocks: split block list
    """
    blocks = []
    amount = len(seq) - (len(seq) % 8)
    for i in range(0, amount, 8):
        block = seq[i: i + 8]
        blocks.append(block)
    logger.info(f"Bits split successfully, result: {blocks}")
    return blocks


def largest_number_of_units(blocks: list) -> dict:
    """The function for finding the maximum sequence length
    Args:
      blocks: split block list
    Returns:
      sorted_dict: statistics of different lengths
    """
    try:
        unit_counts = {}
        for block in blocks:
            cntr = 0
            max_cntr = 0
            for i in block:
                if int(i) == 1:
                    cntr += 1
                    max_cntr = max(max_cntr, cntr)
                    if max_cntr > 4:
                        max_cntr = 4
                else:
                    cntr = 0
            if max_cntr in unit_counts:
                unit_counts[max_cntr] += 1
            else:
                unit_counts[max_cntr] = 1
        sorted_dict = dict(sorted(unit_counts.items(), key=lambda x: x[1]))
        logger.info(f"Dictionary sorted, result: {sorted_dict}")
        return sorted_dict
    except Exception as er:
        logging.error(f"TypeError block wasn't str: {er.message}\n{er.args}\n")


def check_len_test(dictionary: dict) -> float:
    """The function for tests for the longest sequence of units in the block
    Args:
      dictionary: statistics of different lengths
    Returns:
      p_val: probability value for the generator
    """
    try:
        sqr_x = 0
        for i, val in dictionary.items():
            sqr_x += pow(val - 16 * PI[i], 2) / (16 * PI[i])
        p_val = mpmath.gammainc(3 / 2, sqr_x / 2)
        logger.info(f"Longest sequence found, result: {p_val}")
        return p_val
    except Exception as er:
        logging.error(
            f"Length of the dictionary is longer than number of pi-constants: {er.message}\n{er.args}\n"
        )


if __name__ == "__main__":
    print("Значения P-value для числа, сгенерированного на java")
    sequence_java = read_text('D:\SHALAN REP\prog-instruments-labs\lab_4\sequence_java.txt')
    print(frequency_bit_test(sequence_java))
    print(ordinary_bit_test(sequence_java))
    print(
        check_len_test(
            largest_number_of_units(split_bits(sequence_java))
        )
    )
    print("Значения P-value для числа, сгенерированного на cpp")
    sequence_cpp = read_text('D:\SHALAN REP\prog-instruments-labs\lab_4\sequence_cpp.txt')
    print(frequency_bit_test(sequence_cpp))
    print(ordinary_bit_test(sequence_cpp))
    print(
        check_len_test(
            largest_number_of_units(split_bits(sequence_cpp))
        )
    )
