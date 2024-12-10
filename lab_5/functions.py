import math

from scipy.special import gammainc

from constants import P


def read_the_file(input_file_name: str) -> str:
    """
    This function takes the path to the file,
    checks whether it can be opened
    and, if possible, converts the text from the file to a string variable,
    and if impossible, throws an exception

    Args:
        input_file_name (str): a string containing the path to the file

    Raises:
        Exception: an exception appears if the file can not be opened

    Returns:
        str: text from the file
    """
    try:
        text = ""
        with open(input_file_name, "r", encoding="UTF-8") as file:
            text = file.read()
        return text
    except Exception as error:
        raise Exception(f'There is a trouble: {error}')


def frequency_bitwise_test(sequence: str) -> float:
    """
    This function receives a string containing a sequence, 
    after which it counts the sum of its elements (if the element is 1, 
    then it is added as 1 to this sum, if 0 - then as -1), 
    divides by the square root of the number of elements in the sequence, 
    after which the P value is calculated using the special error function erfc

    Args:
        sequence (str): a string containing a sequence of 0 and 1

    Returns:
        float: the P value for the frequency bitwise test
    """
    s = abs(sum(list(map(lambda x: 1 if x == '1' else -1, sequence)))) / \
        math.sqrt(len(sequence))
    return math.erfc(s / math.sqrt(2))


def same_consecutive_bits_test(sequence: str) -> float:
    """
    This function searches for all sequences of identical bits, 
    after which the number and sizes of these sequences are analyzed 
    for compliance with a truly random reference sequence (in short, set the frequency of 1 and 0 shifts)

    Args:
        sequence (str): a string containing a sequence of 0 and 1

    Returns:
        float: the P value for the test of the same consecutive bits
    """
    sequence_length = len(sequence)

    zeta = sum(list(map(lambda x: 1 if x == '1' else 0, sequence))
               ) / sequence_length

    if abs(zeta - 1 / 2) < 2 / math.sqrt(sequence_length):
        v = 0
        for i in range(sequence_length - 1):
            v += 0 if sequence[i] == sequence[i + 1] else 1
        p_value = math.erfc((abs(v - 2 * sequence_length * zeta * (1 - zeta))) /
                      (2 * math.sqrt(2 * sequence_length) * zeta * (1 - zeta)))
    else:
        return 0

    return p_value


def longest_sequence_of_units_in_a_block_test(sequence: str) -> float:
    """
    This function counts the number of blocks with different lengths (<= 1, ==2, ==3, >= 4), 
    calculates the chi-square distribution using the formula, 
    and thanks to it, the P value is calculated for the incomplete gamma function

    Args:
        sequence (str): a string containing a sequence of 0 and 1

    Returns:
        float: the P value for the test of the longest sequence of units in a block
    """
    sequence_length = len(sequence)
    count_of_blocks = sequence_length // 8
    number_of_one_in_the_block = []

    for block in range(count_of_blocks):
        max_ones = 0
        current = 0
        for i in range(8):
            if sequence[8 * block + i] == '1':
                current += 1
                max_ones = max(current, max_ones)
            else:
                current = 0
        number_of_one_in_the_block.append(max_ones)

    v1 = number_of_one_in_the_block.count(
        0) + number_of_one_in_the_block.count(1)
    v2 = number_of_one_in_the_block.count(2)
    v3 = number_of_one_in_the_block.count(3)
    v4 = count_of_blocks - v1 - v2 - v3
    v = [v1, v2, v3, v4]
    xi_distribution = sum(
        math.pow((v[i] - 16 * P[i]), 2) / (16 * P[i]) for i in range(4))

    return gammainc(1.5, xi_distribution / 2)
