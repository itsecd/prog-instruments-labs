import logging
import math
from typing import Optional

from scipy.special import gammaincc

logger = logging.getLogger(__name__)

MIN_BITS_LONGEST_RUN = 128
BLOCK_SIZE = 8
EXPECTED_PROBS = [0.2148, 0.3672, 0.2305, 0.1875]
CHI2_DOF = 3
SIGNIFICANCE = 0.01

_RUN_CATEGORY_UPPER = (1, 2, 3)

INPUT_FILES = [
    ("C++", "generator/C++.txt"),
    ("Java", "generator/Java.txt"),
    ("Python", "generator/Python.txt"),
]

REPORT_PATH = "results.txt"


def read_sequence(filename: str) -> list[int]:
    """Считывает файл и возвращает последовательность цифр 0/1.

    При ошибке чтения возвращает [].
    """
    try:
        with open(filename, "r") as f:
            text = f.read()
    except OSError as e:
        logger.warning("Cannot read %s: %s", filename, e)
        return []
    return [int(c) for c in text if c in "01"]


def frequency_test(bits: list[int]) -> float:
    """Вычисляет p-value для monobit (frequency) test в диапазоне [0, 1]."""
    n = len(bits)
    s_obs = sum(1 if b else -1 for b in bits)
    return math.erfc(abs(s_obs) / math.sqrt(2 * n))


def runs_test(bits: list[int]) -> float:
    """Вычисляет p-value для runs test или 0.0 при сильном перекосе."""
    n = len(bits)
    ones_ratio = sum(bits) / n

    if abs(ones_ratio - 0.5) >= 2 / math.sqrt(n):
        return 0.0

    transitions = 0
    for i in range(n - 1):
        if bits[i] != bits[i + 1]:
            transitions += 1

    numerator = abs(transitions - 2 * n * ones_ratio * (1 - ones_ratio))
    denominator = 2 * math.sqrt(2 * n) * ones_ratio * (1 - ones_ratio)
    return math.erfc(numerator / denominator)


def _max_run_in_block(block: list[int]) -> int:
    """Возвращает максимальную длину серии подряд идущих единиц в блоке."""
    curr = 0
    max_run = 0
    for b in block:
        curr = curr + 1 if b else 0
        max_run = max(max_run, curr)
    return max_run


def _category_index(max_run: int) -> int:
    """Возвращает номер категории для max_run: 0 -> <=1, 1 -> ==2, 2 -> ==3, 3 -> >=4."""
    for i, upper in enumerate(_RUN_CATEGORY_UPPER):
        if max_run <= upper:
            return i
    return len(_RUN_CATEGORY_UPPER)


def _count_by_category(max_runs: list[int]) -> list[int]:
    """Подсчитывает, сколько блоков попало в каждую категорию."""
    category_counts = [0, 0, 0, 0]
    for max_run in max_runs:
        category_counts[_category_index(max_run)] += 1
    return category_counts


def _chi_square_p_value(
    observed: list[int], expected: list[float], dof: int
) -> float:
    """Вычисляет p-value хи-квадрат теста для заданного числа степеней свободы."""
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    return gammaincc(dof / 2, chi2 / 2)


def longest_run_test(bits: list[int]) -> float:
    """Вычисляет p-value для longest-run-of-ones test.

    Требует len(bits) >= 128, иначе возвращает 0.0.
    """
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


def run_tests(bits: list[int]) -> tuple[float, float, float]:
    """Прогоняет все три теста и возвращает тройку p-value."""
    return (
        frequency_test(bits),
        runs_test(bits),
        longest_run_test(bits),
    )


def format_row(
    lang: str,
    freq: Optional[float] = None,
    runs: Optional[float] = None,
    longrun: Optional[float] = None,
) -> str:
    """Формирует одну строку markdown-таблицы.

    Без значений возвращает пустую строку.
    """
    if freq is None:
        return f"| {lang:<11} | {'':<10} | {'':<10} | {'':<10} | {'':<6} |"
    status = "PASSED" if min(freq, runs, longrun) >= SIGNIFICANCE else "FAILED"
    return (
        f"| {lang:<11} | {freq:.7f} | {runs:.7f} | {longrun:.7f} | {status:<6} |"
    )


def build_report(
    input_files: Optional[list[tuple[str, str]]] = None,
) -> list[str]:
    """Формирует список строк отчёта (заголовок и по строке на язык)."""
    files = input_files if input_files is not None else INPUT_FILES
    lines = [
        "| Language    | Frequency | Runs      | LongRun   | Result |",
        "|-------------|-----------|-----------|-----------|--------|",
    ]
    for lang, fname in files:
        bits = read_sequence(fname)
        if bits:
            freq, runs, longrun = run_tests(bits)
            lines.append(format_row(lang, freq, runs, longrun))
        else:
            lines.append(format_row(lang))
    return lines


def write_report(path: str, lines: list[str]) -> None:
    """Записывает строки отчёта в файл."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    lines = build_report()
    write_report(REPORT_PATH, lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
