import math
import numpy as np
from scipy.special import gammaincc

def load_bit_sequence(filepath: str) -> str:
    """
    Функция для чтения битовой последовательности
    """
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read().strip()
    
def frequency_test(filename: str) -> float:
    """
    Частотный тест
    """
    bit_sequence = load_bit_sequence(filename)
    sequence_length = len(bit_sequence)
    
    if sequence_length == 0:
        return 0.0
    
    ones_count = bit_sequence.count("1")
    zeros_count = bit_sequence.count("0")

    deviation = abs(ones_count - zeros_count)
    s_statistic = deviation / math.sqrt(sequence_length)

    p_value = math.erfc(s_statistic / math.sqrt(2))
    return p_value

def bit_test(filename: str) -> float:
    """
    Тест на серии бит
    """
    bit_sequence = load_bit_sequence(filename)
    sequence_length = len(bit_sequence)
    
    ones_probability = bit_sequence.count("1") / sequence_length

    if abs(ones_probability - 0.5) >= (2 / math.sqrt(sequence_length)):
        return 0.0
    
    transitions_count = 0
    for i in range(sequence_length - 1):
        if bit_sequence[i] != bit_sequence[i + 1]:
            transitions_count += 1

    numerator = abs(transitions_count - 2 * sequence_length * ones_probability * (1 - ones_probability))
    denominator = 2 * math.sqrt(2 * sequence_length) * ones_probability * (1 - ones_probability)
    
    p_value = math.erfc(numerator / denominator)
    return p_value

def count_max_runs(filename: str) -> dict:
    """
    Подсчитывает распределение по максимальной длине единиц
    """
    bit_sequence = load_bit_sequence(filename)
    block_size = 8

    runs_distribution = {1: 0, 2: 0, 3: 0, 4: 0}

    for i in range(0, 128, block_size):
        block = bit_sequence[i:i + block_size]

        max_run_length = 0
        current_run_length = 0

        for bit in block:
            if bit == "1":
                current_run_length += 1
                if current_run_length > max_run_length:
                    max_run_length = current_run_length
            else:
                current_run_length = 0

        if max_run_length <= 1:
            runs_distribution[1] += 1
        elif max_run_length == 2:
            runs_distribution[2] += 1
        elif max_run_length == 3:
            runs_distribution[3] += 1
        elif max_run_length >= 4:
            runs_distribution[4] += 1
    
    return runs_distribution

def calculate_chi_square(runs_counts: dict) -> float:
    """
    Вычисляет статистику хи-квадрат для распределения серий
    """
    expected_probabilities = {1: 0.2148, 2: 0.3672, 3: 0.2305, 4: 0.1875}
    chi_square_stat = 0.0
    
    for category in range(1, 5):
        expected_count = 16 * expected_probabilities[category]
        chi_square_stat += ((runs_counts[category] - expected_count) ** 2) / expected_count

    return chi_square_stat

def len_one_test(filename: str) -> float:
    """
    Тест на максимальную серию единиц
    """
    runs_counts = count_max_runs(filename)
    chi_square_stat = calculate_chi_square(runs_counts)

    p_value = gammaincc(3 / 2.0, chi_square_stat / 2.0)
    return p_value

def spectral_test(filename: str) -> float:
    """
    Спектральный тест
    """
    bit_sequence = load_bit_sequence(filename)
    sequence_length = len(bit_sequence)
    
    bipolar_sequence = np.array([2 * int(bit) - 1 for bit in bit_sequence])

    fft_result = np.fft.fft(bipolar_sequence)
    magnitudes = np.abs(fft_result)[:sequence_length // 2]

    threshold = np.sqrt(np.log(1 / 0.05) * sequence_length)

    expected_peaks = 0.95 * sequence_length / 2
    observed_peaks = np.sum(magnitudes < threshold)

    z_score = (observed_peaks - expected_peaks) / np.sqrt(sequence_length * 0.95 * 0.05 / 4)

    p_value = math.erfc(np.abs(z_score) / np.sqrt(2))
    
    return p_value

def main() -> None:
    cpp_output = "output_cpp.txt"
    python_output = "output_py.txt"
    java_output = "output_java.txt"

    print("Тесты:\n")
    
    print("C++:\n")
    print(f"Частотный тест: {frequency_test(cpp_output)}\n")
    print(f"Тест на серии бит: {bit_test(cpp_output)}\n")
    print(f"Тест на максимальную серию единиц: {len_one_test(cpp_output)}\n")
    print(f"Спектральный тест: {spectral_test(cpp_output)}\n")

    print("Python:\n")
    print(f"Частотный тест: {frequency_test(python_output)}\n")
    print(f"Тест на серии бит: {bit_test(python_output)}\n")
    print(f"Тест на максимальную серию единиц: {len_one_test(python_output)}\n")
    print(f"Спектральный тест: {spectral_test(python_output)}\n")

    print("Java:\n")
    print(f"Частотный тест: {frequency_test(java_output)}\n")
    print(f"Тест на серии бит: {bit_test(java_output)}\n")
    print(f"Тест на максимальную серию единиц: {len_one_test(java_output)}\n")
    print(f"Спектральный тест: {spectral_test(java_output)}\n")

if __name__ == "__main__":
    main()