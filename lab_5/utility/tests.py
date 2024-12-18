import math
import mpmath

from utility.constants import PI


def frequency_bitwise_test(sequence: str) -> float:
    """ 
    This function performs a bitwise frequency analysis.
    The test checks how close the P-value is to one.

    Args: 
        sequence (str): sequence of 0 and 1

    Returns: 
        float: the P-value 
    """
    try:    
        sum_seq = sum(list(map(lambda x: 1 if x == '1' else -1, sequence)))
        s = abs(sum_seq) / math.sqrt(len(sequence))
        p_value = math.erfc(s / math.sqrt(2))
        
        return p_value
    except Exception as error:
        print("An error occurred during frequency test:", error)
        

def consecutive_bits_test(sequence: str) -> float:
    """
    This function gets possibility for the identical bits in a sequence.

    Args:
        sequence (str): sequence of 0 and 1

    Returns:
        float: the P-value 
    """
    try:
        seq_len = len(sequence)
        one_c = sequence.count("1") / seq_len
        if abs(one_c - 0.5) > 2 / math.sqrt(seq_len):
            return 0.0
        v = 0
        for i in range(seq_len - 1):
            if sequence[i] != sequence[i + 1]:
                v += 1
        p_value = mpmath.erfc(abs(v - 2 * seq_len * one_c * (1 - one_c)) /
                        (2 * math.sqrt(2 * seq_len) * one_c * (1 - one_c)))
        return p_value
    except Exception as error:
        print("An error occured during consecutive bits test:", error)
        

def longest_sequence_test(sequence: str) -> float:
    """
    This function calculates the P-value for an incomplete gamma function
    by counting the number of blocks of length from 1 to 4
    and calculating the Xi-square

    Args:
        sequence (str): sequence of 0 and 1

    Returns:
        float: the P value for the test of the longest sequence of units in a block
    """
    try:
        blocks = [sequence[i:i + 8] for i in range(0, len(sequence), 8)]
        v = [0, 0, 0, 0]
        for block in blocks:
            max_lenght = 0
            lenght = 0
            for bit in block:
                lenght = lenght + 1 if bit == "1" else 0
                max_lenght = max(max_lenght, lenght)
            match max_lenght:
                case 0 | 1:
                    v[0] += 1
                case 2:
                    v[1] += 1
                case 3:
                    v[2] += 1
                case _:
                    v[3] += 1
        xi = 0
        for i in range(4):
            xi += pow(v[i] - 16 * PI[i], 2) / (16 * PI[i])
        p_value = mpmath.gammainc(3 / 2, xi / 2)

        return p_value
    except Exception as error:
        print("An error occured during longest subseq. test:", error)