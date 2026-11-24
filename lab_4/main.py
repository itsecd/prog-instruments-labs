import math
from scipy.special import gammaincc

MIN_BITS_LONGEST_RUN = 128
BLOCK_SIZE = 8
EXPECTED_PROBS = [0.2148, 0.3672, 0.2305, 0.1875]
CHI2_DOF = 3
SIGNIFICANCE = 0.01

def read_sequence(filename):
    try:
        with open(filename, 'r') as f:
            return [int(c) for c in f.read().strip() if c in '01']
    except:
        return []

def frequency_test(bits):
    n = len(bits)
    s = sum(1 if b else -1 for b in bits)
    return math.erfc(abs(s) / math.sqrt(2 * n))

def runs_test(bits):
    n = len(bits)
    ones_ratio = sum(bits) / n
    if abs(ones_ratio - 0.5) >= 2 / math.sqrt(n):
        return 0.0
    transitions = 0
    for i in range(n - 1):
        if bits[i] != bits[i + 1]:
            transitions += 1
    num = abs(transitions - 2 * n * ones_ratio * (1 - ones_ratio))
    den = 2 * math.sqrt(2 * n) * ones_ratio * (1 - ones_ratio)
    return math.erfc(num / den)

def _max_run_in_block(block):
    curr = max_run = 0
    for b in block:
        curr = curr + 1 if b else 0
        max_run = max(max_run, curr)
    return max_run


def _count_by_category(max_runs):
    category_counts = [0, 0, 0, 0]
    for max_run in max_runs:
        if max_run <= 1:
            category_counts[0] += 1
        elif max_run == 2:
            category_counts[1] += 1
        elif max_run == 3:
            category_counts[2] += 1
        else:
            category_counts[3] += 1
    return category_counts


def _chi_square_p_value(observed, expected, dof):
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    return gammaincc(dof / 2, chi2 / 2)


def longest_run_test(bits):
    n = len(bits)
    if n < MIN_BITS_LONGEST_RUN:
        return 0.0
    num_blocks = n // BLOCK_SIZE
    max_runs = [
        _max_run_in_block(bits[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE])
        for i in range(num_blocks)
    ]
    observed = _count_by_category(max_runs)
    expected = [num_blocks * p for p in EXPECTED_PROBS]
    return _chi_square_p_value(observed, expected, CHI2_DOF)

def main():
    files = [("C++", "generator/C++.txt"), ("Java", "generator/Java.txt"), ("Python", "generator/Python.txt")]
    
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("| Language    | Frequency | Runs      | LongRun   | Result |\n")
        f.write("|-------------|-----------|-----------|-----------|--------|\n")
        
        for lang, fname in files:
            bits = read_sequence(fname)
            if bits:
                freq = frequency_test(bits)
                runs = runs_test(bits)
                longrun = longest_run_test(bits)
                status = "PASSED" if all(p >= 0.01 for p in [freq, runs, longrun]) else "FAILED"
                f.write(f"| {lang:<11} | {freq:.7f} | {runs:.7f} | {longrun:.7f} | {status:<6} |\n")
            else:
                f.write(f"| {lang:<11} | {'':<10} | {'':<10} | {'':<10} | {'':<6} |\n")
    
    with open("results.txt", "r", encoding="utf-8") as f:
        print(f.read())

if __name__ == "__main__":
    main()
