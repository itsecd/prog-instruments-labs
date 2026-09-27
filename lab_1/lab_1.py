import math
from typing import Any, Tuple
import matplotlib.pyplot as plt
import numpy as np


def f_64(x: float) -> np.float64:
    x_val = np.float64(x)
    if x_val == 0.0:
        return np.float64(1.0)
    return (np.exp(x_val) - np.exp(-x_val)) / (np.float64(2.0) * x_val)


def f_32(x: float) -> np.float32:
    x_val = np.float32(x)
    if x_val == np.float32(0.0):
        return np.float32(1.0)
    return (np.exp(x_val) - np.exp(-x_val)) / (np.float32(2.0) * x_val)


def machine_eps(dtype: Any) -> Any:
    eps = dtype(1.0)
    while dtype(1.0 + eps) != dtype(1.0):
        eps /= 2.0
    return eps


def my_function(x: float, eps: float) -> Tuple[float, int]:
    if abs(x) >= 1.0:
        raise ValueError("Ряд расходится! Область сходимости: |x| < 1.")

    term = 1.0
    total = term
    n = 1
    while True:
        ratio = ((2 * n - 1) / (2 * n)) * x
        term = term * ratio
        if abs(term) < eps:
            break
        total += term
        n += 1

    return total, n


def main() -> None:
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №1")
    print("=" * 60)

    header = (
        f"{'k':<5}{'x = 10^-k':<15}{'float64':<25}"
        f"{'float32':<25}{'Относительная погрешность':<20}"
    )
    print(header)
    print("-" * 90)

    for k in range(1, 16):
        x_val = 10.0 ** (-k)
        y64 = f_64(x_val)
        y32 = f_32(x_val)
        rel_err = (
            abs(float(y32) - float(y64)) / abs(float(y64)) if y64 != 0 else 0.0
        )
        print(f"{k:<5}{x_val:<15.1e}{y64:<25.15e}{y32:<25.7e}{rel_err:<20.7e}")

    eps_float32 = machine_eps(np.float32)
    eps_float64 = machine_eps(np.float64)

    print(f"Epsilon (float32): {eps_float32:.8e}")
    print(f"Epsilon (float64): {eps_float64:.16e}")

    test_x_list = [0.2, 0.5, 0.8]
    test_eps_list = [1e-6, 1e-10]
    print(
        f"{'x':<6}{'eps':<10}{'Сумма ряда':<20}"
        f"{'Точное знач. (1/sqrt(1-x))':<30}{'Число слагаемых (n)':<15}"
    )

    for x in test_x_list:
        exact_val = 1.0 / math.sqrt(1.0 - x)
        for eps in test_eps_list:
            approx_val, iters = my_function(x, eps)
            print(
                f"{x:<6.1f}{eps:<10.0e}{approx_val:<30.12f}"
                f"{exact_val:<25.12f}{iters:<15}"
            )

    x_plot = 0.5
    exact_plot = 1.0 / math.sqrt(1.0 - x_plot)
    max_terms = 30
    terms_count = list(range(1, max_terms + 1))
    abs_errors = []
    current_sum = 0.0

    for n in range(max_terms):
        if n == 0:
            term = 1.0
        else:
            term = term * (((2 * n - 1) / (2 * n)) * x_plot)
        current_sum += term
        abs_errors.append(abs(current_sum - exact_plot))

    plt.figure(figsize=(10, 6))
    plt.semilogy(terms_count, abs_errors, marker="o", color="red")
    plt.title(r"График абсолютной погрешности ряда ($x = 0.5$)")
    plt.xlabel("Количество членов ряда ($N$)")
    plt.ylabel("Абсолютная погрешность")
    plt.savefig("error_plot.png")


if __name__ == "__main__":
    main()