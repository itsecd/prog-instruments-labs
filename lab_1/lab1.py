"""Лабораторная работа №1, вариант №4: численные методы и точность вычислений."""

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ZERO_THRESHOLD = 1e-300
MAX_TERMS = 50
TASK4_X = 3.0

Float = np.floating[Any]


def func(x: Float) -> Float:
    """Вычисляет (e^x - 1) / x напрямую.

    Args:
        x: Аргумент в формате float32 или float64.

    Returns:
        Значение функции; при x, близком к нулю, предел 1.
    """
    if abs(x) < ZERO_THRESHOLD:
        return type(x)(1.0)
    return (np.exp(x) - 1) / x


def machine_epsilon(dtype: type[Float]) -> float:
    """Находит машинный эпсилон для заданного типа.

    Args:
        dtype: Тип numpy (np.float32 или np.float64).

    Returns:
        Наименьшее eps, для которого 1 + eps отличимо от 1.
    """
    eps = 1.0
    while dtype(1.0 + eps) != dtype(1.0):
        eps /= 2.0
    return eps * 2.0


def my_func(x: float, eps: float = 1e-10) -> tuple[float, int]:
    """Вычисляет (e^x - 1) / x через ряд Тейлора.

    Args:
        x: Аргумент функции.
        eps: Точность: ряд суммируется, пока член больше eps.

    Returns:
        Кортеж из значения функции и числа учтённых членов ряда.
    """
    if abs(x) < ZERO_THRESHOLD:
        return 1.0, 0

    term = 1.0  # Первый член ряда (x^0 / 0!)
    s = 1.0  # Сумма
    n = 1  # Номер следующего члена

    while abs(term) > eps:
        term *= x / n  # term_n = term_{n-1} * x / n
        s += term
        n += 1

    return (s - 1.0) / x, n - 1


def task1() -> None:
    """Задание 1: сравнение точности float64 и float32."""
    print("\n--- Задание 1: Сравнение float64 и float32 ---")
    print(f"{'x':<10} {'float64':<20} {'float32':<20} {'Отн. ошибка':<15}")
    print("-" * 65)

    for x in np.logspace(-1, -15, 15):
        res64 = func(np.float64(x))
        res32 = func(np.float32(x))
        rel_err = abs(res64 - res32) / abs(res64) if res64 != 0 else 0
        print(f"{x:<10.1e} {res64:<20.10f} {res32:<20.10f} {rel_err:<15.2e}")


def task2() -> None:
    """Задание 2: машинный эпсилон."""
    print("\n--- Задание 2: Машинный эпсилон ---")
    print(f"float64: {machine_epsilon(np.float64):.5e}")
    print(f"float32: {machine_epsilon(np.float32):.5e}")


def task3() -> None:
    """Задание 3: ряд Тейлора для f(x) = (e^x - 1) / x."""
    print("\n--- Задание 3: Ряд Тейлора для f(x) = (e^x - 1)/x ---")

    for x in (0.5, 1, 2):
        for eps in (1e-6, 1e-10):
            approx, terms = my_func(x, eps)
            exact = (math.exp(x) - 1) / x
            error = abs(approx - exact)
            print(
                f"x={x:<3}, eps={eps:.0e}: approx={approx:.10f}, "
                f"exact={exact:.10f}, terms={terms}, error={error:.2e}"
            )


def task4() -> None:
    """Задание 4: график погрешности ряда Тейлора для x=3."""
    print(f"\n--- Задание 4: График погрешности для x={TASK4_X:g} ---")

    exact_val = (math.exp(TASK4_X) - 1) / TASK4_X
    terms_list: list[int] = []
    errors_list: list[float] = []

    term = 1.0
    s = 1.0
    n = 1

    # Накапливаем сумму S для e^x, затем вычисляем (S - 1) / x
    while n < MAX_TERMS:
        term *= TASK4_X / n
        s += term
        n += 1
        approx_val = (s - 1.0) / TASK4_X
        terms_list.append(n - 1)
        errors_list.append(abs(exact_val - approx_val))

    plt.figure(figsize=(10, 6))
    # Полулогарифмический масштаб по Y: видно убывание ошибки
    plt.semilogy(terms_list, errors_list, "o-", color="blue", label="Абс. погрешность")
    plt.title(f"Зависимость погрешности от числа членов ряда (x={TASK4_X:g})")
    plt.xlabel("Число членов ряда")
    plt.ylabel("Абсолютная погрешность")
    plt.legend()
    plt.show()


def main() -> None:
    """Запускает все задания лабораторной работы."""
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №1")
    print("ВАРИАНТ №4")
    print("=" * 60)
    task1()
    task2()
    task3()
    task4()


if __name__ == "__main__":
    main()
