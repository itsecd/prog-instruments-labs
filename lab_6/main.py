import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# Импортируем все функции из нашего файла с логикой
from math_methods import (
    f, df_analytical, left_diff, right_diff, central_diff,
    rectangle_left, rectangle_right, rectangle_mid, trapezoidal, simpson
)

if __name__ == "__main__":
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №6 (Вариант 8)")
    print("=" * 60)

    # --- ЗАДАНИЕ 1: Численное дифференцирование ---
    x0 = 0.5
    h_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    print("\n--- ЗАДАНИЕ 1: Численное дифференцирование ---")
    print(f"x0 = {x0}, точное значение f'(x0) = {df_analytical(x0):.8f}")
    print(f"{'h':<10} {'Левая':<15} {'Правая':<15} {'Центральная':<15} {'Ошибка центр.':<15}")
    print("-" * 75)

    errors_central = []
    for h in h_values:
        dl = left_diff(f, x0, h)
        dr = right_diff(f, x0, h)
        dc = central_diff(f, x0, h)
        err_c = abs(dc - df_analytical(x0))
        errors_central.append(err_c)
        print(f"{h:<10.0e} {dl:<15.8f} {dr:<15.8f} {dc:<15.8f} {err_c:<15.2e}")

    # --- ЗАДАНИЕ 2: Исследование погрешности ---
    plt.figure(figsize=(8, 5))
    plt.loglog(h_values, errors_central, 'o-', color='red', linewidth=2)
    plt.title('Зависимость ошибки центральной разности от шага h')
    plt.xlabel('h (шаг)')
    plt.ylabel('Абсолютная погрешность')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.savefig('lab6_diff_error.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- ЗАДАНИЕ 3: Численное интегрирование ---
    a, b = 0.0, 1.0
    exact = (math.sqrt(math.pi) / 2.0) * math.erf(1.0) 
    n_values = [2, 4, 8, 16, 32, 64]

    print("\n--- ЗАДАНИЕ 3: Численное интегрирование ---")
    print(f"Точное значение интеграла: {exact:.8f}")
    print(f"{'n':<6} {'Левая':<12} {'Правая':<12} {'Средняя':<12} {'Трапеция':<12} {'Симпсон':<12}")
    print("-" * 70)

    for n in n_values:
        il = rectangle_left(f, a, b, n)
        ir = rectangle_right(f, a, b, n)
        im = rectangle_mid(f, a, b, n)
        it = trapezoidal(f, a, b, n)
        isim = simpson(f, a, b, n)
        print(f"{n:<6} {il:<12.6f} {ir:<12.6f} {im:<12.6f} {it:<12.6f} {isim:<12.6f}")

    # --- ЗАДАНИЕ 4: Сравнение погрешностей ---
    errors_left, errors_right, errors_mid, errors_trap, errors_simp = [], [], [], [], []

    for n in n_values:
        errors_left.append(abs(rectangle_left(f, a, b, n) - exact))
        errors_right.append(abs(rectangle_right(f, a, b, n) - exact))
        errors_mid.append(abs(rectangle_mid(f, a, b, n) - exact))
        errors_trap.append(abs(trapezoidal(f, a, b, n) - exact))
        errors_simp.append(abs(simpson(f, a, b, n) - exact))

    plt.figure(figsize=(10, 6))
    plt.loglog(n_values, errors_left, 'o-', label='Левые прямоуг.')
    plt.loglog(n_values, errors_right, 's-', label='Правые прямоуг.')
    plt.loglog(n_values, errors_mid, 'd-', label='Средние прямоуг.')
    plt.loglog(n_values, errors_trap, '^-', label='Трапеции')
    plt.loglog(n_values, errors_simp, '*-', label='Симпсон', linewidth=2)
    plt.title('Сравнение погрешностей интегрирования f(x)=e^(-x^2)')
    plt.xlabel('n')
    plt.ylabel('Абсолютная погрешность')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.savefig('lab6_int_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nГрафики сохранены.")