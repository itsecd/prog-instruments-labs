"""Лабораторная работа №2: нормы матриц, обусловленность, метод Гаусса."""

import matplotlib.pyplot as plt
import numpy as np


def gauss_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решение СЛАУ Ax = b методом Гаусса с выбором главного элемента по столбцу."""
    n = len(a)
    mat = a.astype(float).copy()
    vec = b.astype(float).copy()

    print("\nПрямой ход метода Гаусса:")

    for i in range(n):
        print(f"\nШаг {i + 1}:")
        print(f"Текущая матрица:\n{mat}")
        print(f"Текущий вектор b: {vec}")

        max_row = i
        max_val = abs(mat[i][i])
        for k in range(i + 1, n):
            if abs(mat[k][i]) > max_val:
                max_val = abs(mat[k][i])
                max_row = k

        if max_row != i:
            print(f"Перестановка строк {i + 1} и {max_row + 1}")
            mat[[i, max_row]] = mat[[max_row, i]]
            vec[[i, max_row]] = vec[[max_row, i]]
            print(f"После перестановки:\n{mat}")
            print(f"Вектор b: {vec}")

        if abs(mat[i][i]) < 1e-15:
            raise ValueError(f"Матрица вырождена: главный элемент a[{i}][{i}] = {mat[i][i]}")

        print(f"Главный элемент: {mat[i][i]:.6f}")

        for j in range(i + 1, n):
            factor = mat[j][i] / mat[i][i]
            print(
                f"  Исключение: строка {j + 1} = строка {j + 1} - ({factor:.6f}) * строка {i + 1}"
            )
            mat[j, i:] -= factor * mat[i, i:]
            vec[j] -= factor * vec[i]

    print(f"\nМатрица после прямого хода (верхнетреугольная):\n{mat}")
    print(f"Вектор b после прямого хода: {vec}")

    print("\nОбратный ход:")
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (vec[i] - np.dot(mat[i, i + 1 :], x[i + 1 :])) / mat[i][i]
        print(f"x[{i + 1}] = {x[i]:.6f}")

    return x


def hilbert(n: int) -> np.ndarray:
    """Создание матрицы Гильберта размера n. H[i,j] = 1/(i+j+1)."""
    h = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            h[i, j] = 1.0 / (i + j + 1)
    return h


def main() -> None:
    """Основная функция лабораторной работы."""
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №2")
    print("=" * 60)

    # Задание 1: Нормы матрицы
    a = np.array(
        [
            [5.0, -2.0, 0.0],
            [-2.0, 5.0, -2.0],
            [0.0, -2.0, 5.0],
        ]
    )

    print("\n--- Задание 1: Нормы матрицы ---")
    print("Матрица A:")
    print(a)

    norm_1 = float(np.linalg.norm(a, ord=1))
    norm_inf = float(np.linalg.norm(a, ord=np.inf))
    norm_fro = float(np.linalg.norm(a, ord="fro"))

    print(f"1-норма: {norm_1:.4f}")
    print(f"Бесконечная норма: {norm_inf:.4f}")
    print(f"Норма Фробениуса: {norm_fro:.4f}")

    # Задание 2: Обусловленность
    cond_1 = float(np.linalg.cond(a, p=1))
    cond_inf = float(np.linalg.cond(a, p=np.inf))
    cond_fro = float(np.linalg.cond(a, p="fro"))
    print("\n--- Задание 2: Число обусловленности ---")
    print(f"cond_1 (по 1-норме): {cond_1:.4f}")
    print(f"cond_inf (по бесконечной норме): {cond_inf:.4f}")
    print(f"cond_fro (по норме Фробениуса): {cond_fro:.4f}")

    # Задание 3: Метод Гаусса
    b = np.array([1.0, -1.0, 1.0])

    print("\n--- Задание 3: Решение СЛАУ методом Гаусса ---")
    print("Вектор правой части b:")
    print(b)

    try:
        x_gauss = gauss_solve(a, b)
        print("\nРешение методом Гаусса (с выбором главного элемента):")
        print(x_gauss)

        x_numpy = np.linalg.solve(a, b)
        print("\nРешение встроенным методом numpy.linalg.solve:")
        print(x_numpy)

        error = float(np.linalg.norm(x_gauss - x_numpy, ord=np.inf))
        print(f"\nМаксимальная погрешность (бесконечная норма): {error:.2e}")

        if error < 1e-10:
            print("Решения совпадают с высокой точностью")
        else:
            print("Имеется заметная погрешность")

    except ValueError as e:
        print(f"\nОШИБКА: {e}")

    print("\n" + "=" * 60)

    # Задание 4: Плохо обусловленные матрицы
    print("\n--- Задание 4: Исследование плохо обусловленных матриц ---")

    sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    cond_numbers: list[float] = []

    for n in sizes:
        h = hilbert(n)
        cond = float(np.linalg.cond(h, p=np.inf))
        cond_numbers.append(cond)
        print(f"n = {n:2d}: cond(H) = {cond:.2e}")

    plt.figure(figsize=(8, 5))
    plt.semilogy(sizes, cond_numbers, "o-", color="red", linewidth=2)
    plt.title("Рост числа обусловленности матрицы Гильберта")
    plt.xlabel("Размер матрицы n")
    plt.ylabel("Число обусловленности (логарифм. шкала)")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()
