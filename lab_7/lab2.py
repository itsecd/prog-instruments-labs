"""Лабораторная работа №2 (вариант 4): нормы и обусловленность матриц, метод Гаусса."""

from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

NormOrder = Union[int, float, str]

LINE_WIDTH = 60
SINGULARITY_EPS = 1e-15
NORM_LABEL_WIDTH = 24
COND_LABEL_WIDTH = 39

# Вариант 4
MATRIX_A = np.array([
    [5.0, -2.0, 0.0],
    [-2.0, 5.0, -2.0],
    [0.0, -2.0, 5.0],
])
VECTOR_B = np.array([1.0, -1.0, 1.0])

# (подпись для нормы, подпись для обусловленности, порядок нормы)
NORMS = (
    ("1-норма:", "cond_1 (по 1-норме):", 1),
    ("Бесконечная норма:", "cond_inf (по бесконечной норме):", np.inf),
    ("Норма Фробениуса:", "cond_fro (по норме Фробениуса):", "fro"),
)

HILBERT_SIZES = list(range(2, 11))

PLOT_FIGSIZE = (8, 5)
PLOT_COLOR = "red"
PLOT_LINE_WIDTH = 2
GRID_ALPHA = 0.7


def print_banner(*lines: str) -> None:
    """Печатает строки между двумя горизонтальными линиями."""
    print("=" * LINE_WIDTH)
    for line in lines:
        print(line)
    print("=" * LINE_WIDTH)


def print_task_title(title: str) -> None:
    """Печатает заголовок задания."""
    print(f"\n--- {title} ---")


def hilbert(size: int) -> np.ndarray:
    """Возвращает матрицу Гильберта порядка size."""
    indices = np.arange(size)
    return 1.0 / (np.add.outer(indices, indices) + 1)


def choose_pivot_row(a: np.ndarray, column: int) -> int:
    """Возвращает индекс строки с максимальным по модулю элементом в столбце (от диагонали вниз)."""
    return column + int(np.argmax(np.abs(a[column:, column])))


def swap_rows(a: np.ndarray, b: np.ndarray, first: int, second: int) -> None:
    """Меняет местами строки first и second в матрице a и векторе b."""
    a[[first, second]] = a[[second, first]]
    b[[first, second]] = b[[second, first]]


def forward_elimination(a: np.ndarray, b: np.ndarray) -> None:
    """Прямой ход метода Гаусса с частичным выбором главного элемента (изменяет a и b на месте)."""
    n = len(a)
    for i in range(n):
        pivot_row = choose_pivot_row(a, i)
        if pivot_row != i:
            swap_rows(a, b, i, pivot_row)

        if abs(a[i][i]) < SINGULARITY_EPS:
            raise ValueError("Матрица вырождена (главный элемент близок к нулю)")

        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]
            a[j][i:] -= factor * a[i][i:]
            b[j] -= factor * b[i]


def back_substitution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Обратный ход метода Гаусса для верхнетреугольной матрицы a."""
    n = len(a)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(a[i][i + 1:], x[i + 1:])) / a[i][i]
    return x


def gauss_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Решает СЛАУ Ax = b методом Гаусса с выбором главного элемента по столбцу."""
    a = matrix.astype(float).copy()
    b = rhs.astype(float).copy()
    forward_elimination(a, b)
    return back_substitution(a, b)


def task_matrix_norms(matrix: np.ndarray) -> None:
    """Задание 1: нормы матрицы."""
    print_task_title("Задание 1: Нормы матрицы")
    print("Матрица A:")
    print(matrix)

    for label, _, order in NORMS:
        value = np.linalg.norm(matrix, ord=order)
        print(f"{label:<{NORM_LABEL_WIDTH}}{value:.4f}")


def task_condition_numbers(matrix: np.ndarray) -> None:
    """Задание 2: числа обусловленности матрицы."""
    print_task_title("Задание 2: Число обусловленности")

    for _, label, order in NORMS:
        value = np.linalg.cond(matrix, p=order)
        print(f"{label:<{COND_LABEL_WIDTH}}{value:.4f}")


def task_gauss(matrix: np.ndarray, rhs: np.ndarray) -> None:
    """Задание 3: решение СЛАУ методом Гаусса и сравнение с numpy."""
    print_task_title("Задание 3: Решение СЛАУ методом Гаусса")
    print("Вектор правой части b:")
    print(rhs)

    x_gauss = gauss_solve(matrix, rhs)
    print("Решение методом Гаусса (с выбором главного элемента):")
    print(x_gauss)

    x_numpy = np.linalg.solve(matrix, rhs)
    print("Решение встроенным методом numpy.linalg.solve:")
    print(x_numpy)

    error = np.linalg.norm(x_gauss - x_numpy, ord=np.inf)
    print(f"Максимальная погрешность (бесконечная норма): {error:.2e}")


def plot_condition_growth(sizes: Sequence[int], cond_numbers: Sequence[float]) -> None:
    """Строит график роста числа обусловленности в логарифмической шкале."""
    plt.figure(figsize=PLOT_FIGSIZE)
    plt.semilogy(sizes, cond_numbers, "o-", color=PLOT_COLOR, linewidth=PLOT_LINE_WIDTH)
    plt.title("Рост числа обусловленности матрицы Гильберта")
    plt.xlabel("Размер матрицы n")
    plt.ylabel("Число обусловленности (логарифм. шкала)")
    plt.grid(True, which="both", linestyle="--", alpha=GRID_ALPHA)
    plt.show()


def task_hilbert(sizes: Sequence[int]) -> None:
    """Задание 4: рост числа обусловленности матриц Гильберта."""
    print_task_title("Задание 4: Исследование плохо обусловленных матриц")

    cond_numbers = []
    for size in sizes:
        cond = np.linalg.cond(hilbert(size), p=np.inf)
        cond_numbers.append(cond)
        print(f"n = {size:2d}: cond(H) = {cond:.2e}")

    plot_condition_growth(sizes, cond_numbers)


def main() -> None:
    print_banner("ЛАБОРАТОРНАЯ РАБОТА №2", "Вариант 4")
    task_matrix_norms(MATRIX_A)
    task_condition_numbers(MATRIX_A)
    task_gauss(MATRIX_A, VECTOR_B)
    task_hilbert(HILBERT_SIZES)
    print()
    print_banner("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ")


if __name__ == "__main__":
    main()