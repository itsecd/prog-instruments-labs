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

def longest_run_test(bits):
    n = len(bits)
    if n < MIN_BITS_LONGEST_RUN:
        return 0.0
    num_blocks = n // BLOCK_SIZE
    category_counts = [0, 0, 0, 0]
    ...
    chi2 = sum(
        (category_counts[i] - num_blocks * EXPECTED_PROBS[i]) ** 2
        / (num_blocks * EXPECTED_PROBS[i])
        for i in range(4)
    )
    return gammaincc(CHI2_DOF / 2, chi2 / 2)

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
