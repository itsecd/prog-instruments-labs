import json
import math
import mpmath
from pathlib import Path

from constants import BLOCK_SIZE, PATH, PI_LIST


def read_json(file_path: str) -> dict:
    """Reads the json file"""
    try:
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except:
        return None


def frequency_bitwise_test(seq: str) -> float:
    """Get the results of the Frequency Bitwise test"""
    N = len(seq)
    sum_bits = sum(1 if i == '1' else -1 for i in seq)
    Sn = sum_bits / math.sqrt(N)
    P = math.erfc(Sn/math.sqrt(2))
    return P


def identical_bit_test(seq: str) -> float:
    """Get the results of the Identical Bit test"""
    N = len(seq)
    sum_bits = sum(int(i) for i in seq)
    sig = sum_bits/N
    if abs(sig-0.5) >= 2/math.sqrt(N):
        return 0
    Vn = sum(1 for i in range(len(seq)-1) if seq[i] != seq[i+1])
    P = math.erfc(abs(Vn - 2 * N * sig * (1-sig)) /
                  (2 * math.sqrt(2*N) * sig * (1-sig)))
    return P


def longest_subsequence(sequence: str) -> float:
    """Get the results of the the Longest sequence is 1 in the block test"""
    v = [0, 0, 0, 0]
    for block_start in range(0, len(sequence), BLOCK_SIZE):
        block = sequence[block_start:block_start+BLOCK_SIZE]
        max_length = 0
        for i in range(BLOCK_SIZE, -1, -1):
            if i*"1" in block:
                max_length = i
                break
        match max_length:
            case 0 | 1: 
                v[0] += 1
            case 2:
                v[1] += 1
            case 3:
                v[2] += 1
            case _:
                v[3] += 1
    h_2 = 0
    for i in range(4):
        h_2 += ((v[i]-16*PI_LIST[i])**2/(16*PI_LIST[i]))
    P = mpmath.gammainc(3/2, h_2/2)
    return P


if __name__ == "__main__":
    a = read_json(PATH)
    print(frequency_bitwise_test(a["java"]), identical_bit_test(
        a["java"]), longest_subsequence(a["java"]))
    print(frequency_bitwise_test(
        a["c++"]), identical_bit_test(a["c++"]), longest_subsequence(a["c++"]))
