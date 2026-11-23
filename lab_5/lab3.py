import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №3")
print("Вариант 4: x - cos(x) = 0, x0 = 0.8")
print("=" * 60)

# =====================================================================
# ЗАДАНИЕ 1: Метод Ньютона
# =====================================================================

# Определяем функцию f(x) и её производную f'(x)
# Вариант 4: f(x) = x - cos(x)
def f(x):
    return x - np.cos(x)

def df(x):
    return 1 + np.sin(x)

def newton(f, df, x0, eps=1e-8, max_iter=100):
    """
    Решает уравнение f(x)=0 методом Ньютона.
    x0 - начальное приближение, eps - точность, max_iter - макс. число итераций.
    Возвращает (корень, число итераций, список приближений).
    """
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

# Начальное приближение
x0 = 0.8
root, iterations, hist = newton(f, df, x0)
print("\n--- Задание 1: Метод Ньютона ---")
print(f"Начальное приближение: x0 = {x0}")
print(f"Найденный корень: x = {root:.10f}")
print(f"Число итераций: {iterations}")
print(f"Значение f(x) в корне: {f(root):.2e}")

# =====================================================================
# ЗАДАНИЕ 2: Метод простых итераций
# =====================================================================

# Приводим уравнение к виду x = phi(x).
# x - cos(x) = 0  =>  x = cos(x),  phi(x) = cos(x).
# Условие сходимости: |phi'(x)| = |sin(x)| = sin(x) <= sin(1) ~ 0.84 < 1
# на отрезке [0, 1], содержащем корень. Релаксация не нужна.
def phi(x):
    return np.cos(x)

def fixed_point(phi, x0, eps=1e-8, max_iter=2000):
    """
    Решает уравнение x = phi(x) методом простых итераций.
    Возвращает (корень, число итераций, список приближений) или
    (None, число итераций, history), если метод не сошёлся.
    """
    x = x0
    history = [x]
    for i in range(max_iter):
        x_new = phi(x)
        history.append(x_new)
        # защита от расходимости (переполнения)
        if not np.isfinite(x_new) or abs(x_new) > 1e10:
            return None, i + 1, history
        if abs(x_new - x) < eps:
            return x_new, i + 1, history
        x = x_new
    return None, max_iter, history

# Начальное приближение (то же, что для метода Ньютона)
x0_fixed = 0.8
root_fixed, iter_fixed, hist_fixed = fixed_point(phi, x0_fixed)
print("\n--- Задание 2: Метод простых итераций ---")
print("Итерационная функция: phi(x) = cos(x)")
print(f"Начальное приближение: x0 = {x0_fixed}")
if root_fixed is None:
    print(f"Метод НЕ сошёлся за {iter_fixed} итераций")
    print(f"Последнее приближение: x ~ {hist_fixed[-1]:.6e}")
else:
    print(f"Найденный корень: x = {root_fixed:.10f}")
    print(f"Число итераций: {iter_fixed}")
    print(f"Значение f(x) в корне: {f(root_fixed):.2e}")
    print(f"Сравнение: Ньютон - {iterations} ит., простые итерации - {iter_fixed} ит.")

# =====================================================================
# ЗАДАНИЕ 3: Сравнение методов
# =====================================================================

print("\n--- Задание 3: Сравнение методов ---")
print(f"{'Метод':<20} {'Нач. прибл.':<12} {'Итераций':<10} {'Корень':<15} {'Невязка':<12}")
print("-" * 70)

for x0_t in [x0 - 0.5, x0, x0 + 0.5]:
    # Метод Ньютона
    try:
        root_n, it_n, _ = newton(f, df, x0_t, eps=1e-10)
        root_n_str = f"{root_n:.8f}"
        res_n_str = f"{f(root_n):.2e}"
        it_n_str = str(it_n)
    except Exception:
        root_n_str, it_n_str, res_n_str = "не сошёлся", "-", "-"

    # Метод простых итераций
    root_f, it_f, _ = fixed_point(phi, x0_t, eps=1e-10)
    if root_f is None:
        root_f_str, res_f_str, it_f_str = "не сошёлся", "-", str(it_f)
    else:
        root_f_str = f"{root_f:.8f}"
        res_f_str = f"{f(root_f):.2e}"
        it_f_str = str(it_f)

    print(f"{'Ньютон':<20} {x0_t:<12.2f} {it_n_str:<10} {root_n_str:<15} {res_n_str:<12}")
    print(f"{'Простые итерации':<20} {x0_t:<12.2f} {it_f_str:<10} {root_f_str:<15} {res_f_str:<12}")

# =====================================================================
# ЗАДАНИЕ 4: Визуализация сходимости
# =====================================================================

print("\n--- Задание 4: График сходимости ---")
x_vals = np.linspace(0, 1.5, 400)
y_vals = f(x_vals)

# График 1: функция и итерации Ньютона
plt.figure(figsize=(9, 6))
plt.plot(x_vals, y_vals, label='f(x) = x - cos(x)', linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(root, color='red', linestyle='--',
            label=f'Корень x = {root:.4f}')
plt.plot(hist, [f(xi) for xi in hist], 'ro-', markersize=6,
         linewidth=1.2, label='Итерации Ньютона')
plt.title('Метод Ньютона: поиск корня уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.6)
plt.savefig('lab3_fig1.png', dpi=150, bbox_inches='tight')
plt.show()

# График 2: сходимость x_n -> x*
plt.figure(figsize=(9, 5))
plt.plot(range(len(hist)), hist, 'o-', color='darkred',
         linewidth=1.5, label='Метод Ньютона')
plt.plot(range(len(hist_fixed)), hist_fixed, 's--', color='darkblue',
         linewidth=1.5, label='Простые итерации (phi = cos x)')
plt.axhline(root, color='green', linestyle=':', linewidth=1.5,
            label=f'Корень x* = {root:.6f}')
plt.title('Сравнение сходимости методов')
plt.xlabel('Номер итерации')
plt.ylabel('Приближение x_n')
plt.xlim(-0.5, 40)
plt.legend()
plt.grid(True, alpha=0.6)
plt.savefig('lab3_fig2.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ")
print("=" * 60)
