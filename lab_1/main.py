from collections.abc import Callable
from math import isfinite

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# ЗАДАНИЕ 1: Метод Ньютона
# ============================================================
# f(x) = x^3 + x - 1, f'(x) = 3x^2 + 1
def f(x: float) -> float:
    return x**3 + x - 1


def df(x: float) -> float:
    return 3 * x**2 + 1


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    eps: float = 1e-8,
    max_iter: int = 100,
) -> tuple[float, int, list[float]]:
    """Метод Ньютона. Возвращает (корень, число итераций, список приближений)."""
    x = x0
    history = [x]
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            raise ValueError("Производная равна нулю (метод Ньютона не применим)")
        x_new = x - fx / dfx
        history.append(x_new)
        if abs(x_new - x) < eps or abs(f(x_new)) < eps:
            return x_new, i + 1, history
        x = x_new
    raise RuntimeError("Метод Ньютона не сошёлся за заданное число итераций")


# ============================================================
# ЗАДАНИЕ 2: Метод простых итераций
# ============================================================
# x = phi(x) = x - tau*f(x). Условие сходимости: |1 - tau*f'(x*)| < 1.
# f'(x*) ~ 2.397 -> tau = 0.3 даёт q = |1 - 0.3*2.397| ~ 0.28
TAU = 0.3


def phi(x: float) -> float:
    return x - TAU * f(x)


def fixed_point(
    phi: Callable[[float], float], x0: float, eps: float = 1e-8, max_iter: int = 2000
) -> tuple[float | None, int, list[float]]:
    """Метод простых итераций. Возвращает (корень | None, итерации, history)."""
    x = x0
    history = [x]
    for i in range(max_iter):
        x_new = phi(x)
        history.append(x_new)
        if not isfinite(x_new) or abs(x_new) > 1e10:
            return None, i + 1, history
        if abs(x_new - x) < eps:
            return x_new, i + 1, history
        x = x_new
    return None, max_iter, history


def main() -> None:
    matplotlib.use("Agg")
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №3. Вариант 13: x^3 + x - 1 = 0, x0 = 0.5")
    print("=" * 60)

    x0 = 0.5
    root, iterations, hist = newton(f, df, x0)
    print("\n--- Задание 1: Метод Ньютона ---")
    print(f"Начальное приближение: x0 = {x0}")
    print(f"Найденный корень: x = {root:.10f}")
    print(f"Число итераций: {iterations}")
    print(f"Значение f(x) в корне: {f(root):.2e}")

    x0_fixed = 0.5
    root_fixed, iter_fixed, hist_fixed = fixed_point(phi, x0_fixed)
    print("\n--- Задание 2: Метод простых итераций ---")
    print(f"Параметр релаксации: tau = {TAU}")
    print(f"Коэффициент сходимости q = |phi'(x*)| = {abs(1 - TAU * df(root)):.4f}")
    print(f"Начальное приближение: x0 = {x0_fixed}")
    if root_fixed is None:
        print(f"Метод НЕ сошёлся за {iter_fixed} итераций")
        print(f"Последнее приближение: x ~ {hist_fixed[-1]:.6e}")
    else:
        print(f"Найденный корень: x = {root_fixed:.10f}")
        print(f"Число итераций: {iter_fixed}")
        print(f"Значение f(x) в корне: {f(root_fixed):.2e}")

    # ============================================================
    # ЗАДАНИЕ 3: Сравнение методов
    # ============================================================
    print("\n--- Задание 3: Сравнение методов ---")
    print(f"{'Метод':<20} {'Нач. прибл.':<12} {'Итераций':<10} {'Корень':<15} {'Невязка':<12}")
    print("-" * 70)

    for x0_ in [0.0, 0.5, 1.0]:
        try:
            root_n, it_n, _ = newton(f, df, x0_, eps=1e-10)
            root_n_str = f"{root_n:.8f}"
            res_n_str = f"{f(root_n):.2e}"
            it_n_str = str(it_n)
        except (ValueError, RuntimeError):
            root_n_str, it_n_str, res_n_str = "не сошёлся", "-", "-"

        root_f, it_f, _ = fixed_point(phi, x0_, eps=1e-10)
        if root_f is None:
            root_f_str, res_f_str, it_f_str = "не сошёлся", "-", str(it_f)
        else:
            root_f_str = f"{root_f:.8f}"
            res_f_str = f"{f(root_f):.2e}"
            it_f_str = str(it_f)

        print(f"{'Ньютон':<20} {x0_:<12.2f} {it_n_str:<10} {root_n_str:<15} {res_n_str:<12}")
        print(
            f"{'Простые итерации':<20} {x0_:<12.2f} {it_f_str:<10} {root_f_str:<15} {res_f_str:<12}"
        )

    # ============================================================
    # ЗАДАНИЕ 4: Визуализация сходимости
    # ============================================================
    print("\n--- Задание 4: График сходимости ---")
    x_vals = np.linspace(0, 1.2, 400)
    y_vals = [f(float(x)) for x in x_vals]

    plt.figure(figsize=(9, 6))
    plt.plot(x_vals, y_vals, label="f(x) = x^3 + x - 1", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(root, color="red", linestyle="--", label=f"Корень x = {root:.4f}")
    plt.plot(
        hist, [f(xi) for xi in hist], "ro-", markersize=6, linewidth=1.2, label="Итерации Ньютона"
    )
    plt.title("Метод Ньютона: поиск корня уравнения")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.savefig("fig1_newton.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(range(len(hist)), hist, "o-", color="darkred", linewidth=1.5, label="Метод Ньютона")
    plt.plot(
        range(len(hist_fixed)),
        hist_fixed,
        "s--",
        color="darkblue",
        linewidth=1.5,
        label=f"Простые итерации (tau={TAU})",
    )
    plt.axhline(root, color="green", linestyle=":", linewidth=1.5, label=f"Корень x* = {root:.6f}")
    plt.title("Сравнение сходимости методов")
    plt.xlabel("Номер итерации")
    plt.ylabel("Приближение x_n")
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.savefig("fig2_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 60)
    print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ")
    print("=" * 60)

    # --- дополнительно для отчёта: погрешности по итерациям ---
    print("\nПогрешности |x_n - x*|:")
    print("Ньютон:", [f"{abs(h - root):.2e}" for h in hist])
    print("Простые итерации (первые 12):", [f"{abs(h - root):.2e}" for h in hist_fixed[:12]])


if __name__ == "__main__":
    main()
