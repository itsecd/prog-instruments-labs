from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def f(x: Any) -> Any:
    return 1.0 / (1.0 + 16.0 * (x**2))


def lagrange(x: float, x_nodes: Any, y_nodes: Any) -> float:
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        li = 1.0
        for j in range(n):
            if j != i:
                li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * li
    return float(result)


def least_squares(x_nodes: Any, y_nodes: Any, degree: int) -> Any:
    m = degree + 1
    A = np.zeros((m, m))
    b = np.zeros(m)
    for i in range(m):
        for j in range(m):
            A[i, j] = np.sum(x_nodes ** (i + j))
        b[i] = np.sum(y_nodes * (x_nodes**i))
    coeff = np.linalg.solve(A, b)
    return coeff


def approx_poly(x: Any, coeff: Any) -> Any:
    return np.polyval(coeff[::-1], x)


def main() -> None:
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №5. ВАРИАНТ 12")
    print("=" * 60)

    n_base = 5
    degree_base = 2
    x_nodes = np.linspace(-1.0, 1.0, n_base)
    y_nodes = f(x_nodes)

    x_extended = np.arange(x_nodes[0] - 0.5, x_nodes[-1] + 0.50001, 0.05)
    y_lagrange_ext = [lagrange(xi, x_nodes, y_nodes) for xi in x_extended]

    print("\n--- Задание 1: Значения полинома Лагранжа (шаг 0.05, фрагмент) ---")
    for xi, yi in zip(x_extended[::5], y_lagrange_ext[::5]):
        print(f"x = {xi:6.2f} | P_Lagr(x) = {yi:10.4f}")

    coeff = least_squares(x_nodes, y_nodes, degree_base)
    print(f"\n--- Задание 2: Коэффициенты МНК (степень k = {degree_base}) ---")
    print(f"g(x) = {coeff[0]:.4f} + ({coeff[1]:.4e})*x + ({coeff[2]:.4f})*x^2")

    print("\n--- Задание 3: Сравнение в узлах ---")
    print(
        f"{'x':<8} {'y (исх.)':<12} {'Интерполяция':<15} {'Погр. инт.':<12} {'Аппроксимация':<15} {'Погр. аппр.':<12}"
    )
    print("-" * 76)
    y_lagr_nodes = [lagrange(xi, x_nodes, y_nodes) for xi in x_nodes]
    y_appr_nodes = approx_poly(x_nodes, coeff)
    for i in range(n_base):
        err_l = abs(y_lagr_nodes[i] - y_nodes[i])
        err_a = abs(y_appr_nodes[i] - y_nodes[i])
        print(
            f"{x_nodes[i]:<8.2f} {y_nodes[i]:<12.4f} {y_lagr_nodes[i]:<15.4f} {err_l:<12.2e} {y_appr_nodes[i]:<15.4f} {err_a:<12.4f}"
        )

    x_plot = np.linspace(-1.0, 1.0, 300)
    y_true = f(x_plot)
    y_plot_lagr = [lagrange(xi, x_nodes, y_nodes) for xi in x_plot]
    y_plot_appr = approx_poly(x_plot, coeff)

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_true, "k-", linewidth=2, label="Точная функция f(x)")
    plt.plot(x_nodes, y_nodes, "ko", markersize=8, label="Узлы (n = 5)")
    plt.plot(
        x_plot, y_plot_lagr, "r-", linewidth=1.8, label="Интерполяция Лагранжа (deg=4)"
    )
    plt.plot(
        x_plot,
        y_plot_appr,
        "b--",
        linewidth=2,
        label=f"МНК аппроксимация (deg={degree_base})",
    )
    plt.title("Интерполяция и аппроксимация f(x) = 1/(1+16x^2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    n_values = [5, 8, 11, 15]
    plt.figure(figsize=(9, 5.5))
    plt.plot(x_plot, y_true, "k-", linewidth=2.5, label="Точная f(x)")
    styles = ["g--", "c-.", "m:", "r-"]
    for n_val, st in zip(n_values, styles):
        xn = np.linspace(-1.0, 1.0, n_val)
        yn = f(xn)
        y_int = [lagrange(xi, xn, yn) for xi in x_plot]
        max_e = np.max(np.abs(np.array(y_int) - y_true))
        plt.plot(
            x_plot,
            y_int,
            st,
            linewidth=1.8,
            label=f"Интерполяция n={n_val} (max err={max_e:.2f})",
        )

    plt.title("Феномен Рунге для функции f(x) = 1/(1+16x^2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(-0.5, 2.5)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
