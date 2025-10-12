import time
import numpy as np
from IB import dataset
from optimized_IB import fast_multivariate_normal_pdf, add_optimized_methods
from data_generation import gen_easytest


def test_basic_optimization():
    """Простой тест векторизованного PDF"""
    print("=== Testing Basic PDF Optimization ===")

    # Тестовые данные
    grid_points = np.random.rand(100, 2)  # 100 точек сетки
    means = np.random.rand(5, 2)  # 5 средних значений
    cov = np.eye(2) * 0.1  # Одна ковариационная матрица

    # Тестируем скорость
    start = time.time()
    result_fast = fast_multivariate_normal_pdf(grid_points, means, cov)
    time_fast = time.time() - start

    # Сравниваем с оригинальным подходом
    from scipy.stats import multivariate_normal
    start = time.time()
    result_original = np.zeros((100, 5))
    for i in range(5):
        rv = multivariate_normal(means[i], cov)
        result_original[:, i] = rv.pdf(grid_points)
    time_original = time.time() - start

    print(f"Original loop: {time_original:.4f}s")
    print(f"Vectorized:    {time_fast:.4f}s")
    print(f"Speedup: {time_original / time_fast:.1f}x")

    # Проверяем корректность
    diff = np.max(np.abs(result_original - result_fast))
    print(f"Max difference: {diff:.10f}")

    return time_original / time_fast


def integrate_optimization():
    """Интегрируем оптимизацию в существующий код"""
    print("\n=== Integrating with IB Code ===")

    # Добавляем оптимизированные методы в класс dataset
    add_optimized_methods()

    # Создаем небольшой dataset как в профайлинге
    ds = gen_easytest(plot=False)
    ds.s = 0.5
    ds.smoothing_type = 'uniform'

    # Замеряем оригинальную версию
    start = time.time()
    ds.coord_to_pxy(total_bins=100)
    time_original = time.time() - start
    original_ixy = ds.ixy
    print(f"Original coord_to_pxy: {time_original:.3f}s, I(X;Y)={original_ixy:.3f}")

    # Теперь с оптимизацией
    ds2 = gen_easytest(plot=False)
    ds2.s = 0.5
    ds2.smoothing_type = 'uniform'

    start = time.time()
    ds2.fast_coord_to_pxy(total_bins=100)  # Новый оптимизированный метод
    time_optimized = time.time() - start
    optimized_ixy = ds2.ixy
    print(f"Optimized coord_to_pxy: {time_optimized:.3f}s, I(X;Y)={optimized_ixy:.3f}")

    print(f"Speedup: {time_original / time_optimized:.1f}x")
    print(f"I(X;Y) difference: {abs(original_ixy - optimized_ixy):.6f}")


if __name__ == "__main__":
    speedup = test_basic_optimization()

    if speedup > 2:  # Если ускорение значительное
        integrate_optimization()
    else:
        print("Optimization needs more work before integration")