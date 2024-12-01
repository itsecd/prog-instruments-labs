import math
import logging
import mpmath
import file_readers


def frequency_bit_test(sequence: str) -> float:
    """
    This function gets possibility for the frequency bit test
    
    Parametres:
        sequence (str): sequence of bytes
    
    Returns:
        float: P value
    """
    N = len(sequence)
    sum_bits = sum(1 if i == '1' else -1 for i in sequence)
    P = sum_bits / math.sqrt(N)
    P = math.erfc(P/math.sqrt(2))
    return P


def identical_bit_test(sequence: str) -> float:
    """
    This function gets possibility for the identical bits in a sequence

    Parametres:
        sequence (str): sequence of bytes

    Returns:
        float: P value
    """
    N = len(sequence)
    sum_bits = sum(int(i) for i in sequence)
    sig = sum_bits/N
    if abs(sig-0.5)>=2/math.sqrt(2): return 0
    Vn = sum(1 for i in range(len(sequence)-1) if sequence[i] != sequence[i+1])
    P = math.erfc(abs(Vn-2*N*sig*(1-sig))/(2*math.sqrt(2*N)* sig *(1-sig)))
    return P


def longest_subsequence(sequence: str, consts_PI: list) -> float:
    """
    This function gets possibility for the longest sequence of ones in a block

    Parametres:
        sequence (str): sequence of bytes
        consts_PI (list): list of constants for the tests
    
    Returns:
        float: P value
    """
    block_length = 8
    v = {1: 0, 2: 0, 3:0, 4: 0}
    hi_2 = 0
    for block_start in range(0, len(sequence), block_length):
        block = sequence[block_start:block_start+block_length]
        max_length, length = 0, 0
        for bit in block:
            if bit == "1":
                length += 1
                max_length = max(max_length, length)
            else:
                length = 0
        match max_length:
            case 0 | 1: 
                v[1] += 1
            case 2:
                v[2] += 1
            case 3:
                v[3] += 1
            case _:
                v[4] += 1
    for i in range(4):
        hi_2 += ((v[i+1]-16*consts_PI[i])**2/(16*consts_PI[i]))
    P = mpmath.gammainc(3/2, hi_2/2)
    return P


def run_test_and_write(sequence: str, output_file: str, consts_PI: list):
    """
    This function runs the tests and writes the results into the output file

    Parametres:
        sequence(str): sequence of bytes
        output_file(str): output file name
        consts_PI(list): constants for the test
    """
    freq_result = frequency_bit_test(sequence)
    ident_result = identical_bit_test(sequence)
    longest_result = longest_subsequence(sequence, consts_PI)

    results = [f"Frequency Bit Test: {freq_result}",f"Identical Bit Test: {ident_result}", f"Longest Subsequence Test: {longest_result}"]
    file_readers.write_file(output_file, results)


if __name__ == "__main__":
    settings = file_readers.read_json_file("settings.json")
    sequences = file_readers.read_json_file("generated.json")
    consts_PI = settings["consts_PI"]
    input_c = sequences["cpp"]
    input_java = sequences["java"]
    output_file_c = settings["c_output"]
    output_file_java = settings["java_output"]
    run_test_and_write(input_c, output_file_c, consts_PI)
    run_test_and_write(input_java, output_file_java, consts_PI)