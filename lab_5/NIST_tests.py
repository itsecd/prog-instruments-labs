import math
import os
import json
import logging


from typing import List, Union
from r_files import read_txt_file
from scipy.special import gammaincc


PI: List[float] = [0.2148, 0.3672, 0.2305, 0.1875]


def frequency_test(bits: str) -> Union[float, None]:
    """
    Performs the frequency test on a binary sequence.

    Args:
        bits: The binary sequence.

    Returns:
        The p-value of the test, or None if an error occurs.
    """
    try:
        n = len(bits)
        s_obs = 0

        for bit in bits:
            if bit == '1':
                s_obs += 1
            else:
                s_obs -= 1

        Sn = (1 / math.sqrt(n)) * s_obs
        P_value = math.erfc(Sn / math.sqrt(2))
        
        return P_value
    except Exception as e:
        logging.error(f"Frequency test failed: {e}")
        return None


def consecutive_bits_test(bits: str) -> Union[float, None]:
    """
    Performs the consecutive bits test on a binary sequence.

    Args:
        bits: The binary sequence.

    Returns:
        The p-value of the test, or None if an error occurs.
    """
    try:
        n = len(bits)
        ones_count = bits.count('1')
        e = ones_count / n
        
        if not abs(e - 0.5) < (2 / math.sqrt(n)):
            return 0
        
        Vn = 0
        for i in range(1, n):
            if bits[i] != bits[i - 1]:
                Vn += 1

        P_value = math.erfc(abs(Vn - 2 * n * e * (1 - e)) / (2 * math.sqrt(2 * n) * e * (1 - e)))
        
        return P_value
    except Exception as e:
        logging.error(f"Consecutive bits test failed: {e}")
        return None


def longest_run_of_ones_test(bits: str) -> Union[float, None]:
    """
    Performs the longest run of ones test on a binary sequence.

    Args:
        bits: The binary sequence.

    Returns:
        The p-value of the test, or None if an error occurs.
    """
    try:
        n = len(bits)
        block_size = 8
        num_blocks = n // block_size
        max_run_lengths = [0] * num_blocks
        for i in range(num_blocks):
            block = bits[i * block_size: (i + 1) * block_size]
            max_run = 0
            current_run = 0

            for bit in block:
                if bit == '1':
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0

            max_run_lengths[i] = max_run

        freq_count = [0, 0, 0, 0]
        for length in max_run_lengths:
            match length:
                case 0 | 1:
                    freq_count[0] += 1
                case 2:
                    freq_count[1] += 1
                case 3:
                    freq_count[2] += 1
                case 4 | 5 | 6 | 7 | 8:
                    freq_count[3] += 1


        chi_squared = sum([(freq_count[i] - 16 * PI[i]) ** 2 / 16 * PI[i] for i in range(len(PI))])
        P_value = gammaincc(3 / 2, chi_squared / 2)

        return P_value
    except Exception as e:
        logging.error(f"Longest run of ones test failed: {e}")
        return None


def main() -> None:
    """
    Executes the main logic of the program.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        c_path = os.path.join(script_dir, 'config.json')
            
        with open(c_path, 'r') as config_file:
            config = json.load(config_file)


        # Read input files
        cpp = config["input_cpp"]
        input_cpp = read_txt_file(cpp)
        java = config["input_java"]
        input_java = read_txt_file(java)

        cpp_results = {
            'Frequency Test': frequency_test(input_cpp),
            'Consecutive Bits Test': consecutive_bits_test(input_cpp),
            'Longest Run of Ones Test': longest_run_of_ones_test(input_cpp)
        }

        java_results = {
            'Frequency Test': frequency_test(input_java),
            'Consecutive Bits Test': consecutive_bits_test(input_java),
            'Longest Run of Ones Test': longest_run_of_ones_test(input_java)
        }

        # Write results to output file
        with open(config['output'], 'w') as output_file:
            output_file.write("Results for C++ Generator:\n")
            for test, result in cpp_results.items():
                if result is not None:
                    output_file.write(f"{test}: {result}\n")

            output_file.write("\nResults for Java Generator:\n")
            for test, result in java_results.items():
                if result is not None:
                    output_file.write(f"{test}: {result}\n")
    except Exception as e:
        logging.error(f"An error occurred in main(): {e}")


if __name__ == "__main__":
    main()