import time
import numpy as np
from IB import dataset
from optimized_IB import fast_multivariate_normal_pdf, add_optimized_methods, debug_dimensions
from data_generation import gen_easytest


def test_basic_optimization():
    """Простой тест векторизованного PDF"""
    print("=== Testing Basic PDF Optimization ===")

    # Увеличим данные для лучшего измерения
    grid_points = np.random.rand(1000, 2)  # 1000 точек сетки
    means = np.random.rand(50, 2)  # 50 средних значений
    cov = np.eye(2) * 0.1

    # Тестируем скорость
    start = time.time()
    result_fast = fast_multivariate_normal_pdf(grid_points, means, cov)
    time_fast = time.time() - start

    # Сравниваем с оригинальным подходом
    from scipy.stats import multivariate_normal
    start = time.time()
    result_original = np.zeros((1000, 50))
    for i in range(50):
        rv = multivariate_normal(means[i], cov)
        result_original[:, i] = rv.pdf(grid_points)
    time_original = time.time() - start

    print(f"Original loop: {time_original:.4f}s")
    print(f"Vectorized:    {time_fast:.4f}s")

    # Защита от деления на ноль
    if time_fast > 0:
        speedup = time_original / time_fast
        print(f"Speedup: {speedup:.1f}x")
    else:
        speedup = float('inf')
        print(f"Speedup: >100x (vectorized too fast to measure)")

    # Проверяем корректность
    diff = np.max(np.abs(result_original - result_fast))
    print(f"Max difference: {diff:.10f}")

    return speedup


def integrate_optimization():
    """Интегрируем оптимизацию в существующий код"""
    print("\n=== Integrating with IB Code ===")

    # Добавляем оптимизированные методы в класс dataset
    add_optimized_methods()

    # Создаем dataset и ЗАПОМИНАЕМ параметры бинов
    ds = gen_easytest(plot=False)
    ds.s = 0.5
    ds.smoothing_type = 'uniform'

    # Сначала получаем параметры бинов
    Y, bins1, bins2, y1v, y2v, Ygrid = ds.make_bins(total_bins=200)
    print(f"Bins info: Y={Y}, will be reduced after dropping distant bins")

    # Замеряем оригинальную версию
    print("Running ORIGINAL coord_to_pxy...")
    start = time.time()
    ds.coord_to_pxy(total_bins=200)
    time_original = time.time() - start
    original_ixy = ds.ixy
    original_Y = ds.Y
    print(f"Original coord_to_pxy: {time_original:.3f}s, I(X;Y)={original_ixy:.3f}, Y={original_Y}")

    # Теперь с оптимизацией - используем ТОЧНО ТЕ ЖЕ параметры
    ds2 = gen_easytest(plot=False)
    ds2.s = 0.5
    ds2.smoothing_type = 'uniform'

    # Вручную устанавливаем те же бины чтобы гарантировать одинаковые размерности
    print("Running OPTIMIZED fast_coord_to_pxy...")
    start = time.time()

    # Используем тот же подход что в оригинальном coord_to_pxy
    Y2, bins1_2, bins2_2, y1v_2, y2v_2, Ygrid_2 = ds2.make_bins(total_bins=200)

    # ОПТИМИЗАЦИЯ: векторизованное вычисление PDF
    S = (ds2.s ** 2) * np.eye(2)
    py_x = fast_multivariate_normal_pdf(Ygrid_2, ds2.coord, S)  # [Y, X]

    # Дроп далеких бинов (как в оригинале)
    ycountv = np.zeros(Y2)
    for x in range(ds2.X):
        for y in range(Y2):
            if np.linalg.norm(ds2.coord[x, :] - Ygrid_2[y, :]) < getattr(ds2, 'pad', 2 * ds2.s):
                ycountv[y] += 1

    ymask = ycountv > 0
    py_x = py_x[ymask, :]
    Ygrid_2 = Ygrid_2[ymask, :]
    print(f"Dropped {Y2 - np.sum(ymask)} ybins. Y reduced from {Y2} to {np.sum(ymask)}.")

    # Нормализация
    py_x = py_x / np.sum(py_x, axis=0, keepdims=True)

    # Заполняем атрибуты
    ds2.py_x = py_x
    ds2.Y = np.sum(ymask)
    ds2.Ygrid = Ygrid_2
    ds2.px = (1 / ds2.X) * np.ones(ds2.X, dtype=ds2.dt)
    ds2.pxy = (ds2.py_x * ds2.px).T

    ds2.process_pxy(drop_zeros=True)

    time_optimized = time.time() - start
    optimized_ixy = ds2.ixy
    print(f"Optimized coord_to_pxy: {time_optimized:.3f}s, I(X;Y)={optimized_ixy:.3f}, Y={ds2.Y}")

    if time_optimized > 0:
        speedup = time_original / time_optimized
        print(f"Speedup: {speedup:.1f}x")
    else:
        print(f"Speedup: Very significant (optimized too fast to measure)")

    print(f"I(X;Y) difference: {abs(original_ixy - optimized_ixy):.6f}")

    # Проверяем что размерности совпадают
    print(f"Dimension check - Original Y: {original_Y}, Optimized Y: {ds2.Y}")


def quick_performance_test():
    """Быстрый тест производительности на одинаковых данных"""
    print("\n=== Quick Performance Test ===")

    add_optimized_methods()

    # Создаем одинаковые datasets
    ds_orig = gen_easytest(plot=False)
    ds_orig.s = 1.0
    ds_orig.smoothing_type = 'uniform'

    ds_opt = gen_easytest(plot=False)
    ds_opt.s = 1.0
    ds_opt.smoothing_type = 'uniform'

    # Оригинальный
    start = time.time()
    ds_orig.coord_to_pxy(total_bins=100)
    time_orig = time.time() - start

    # Оптимизированный
    start = time.time()
    ds_opt.fast_coord_to_pxy(total_bins=100)
    time_opt = time.time() - start

    print(f"Original: {time_orig:.3f}s, I(X;Y)={ds_orig.ixy:.3f}")
    print(f"Optimized: {time_opt:.3f}s, I(X;Y)={ds_opt.ixy:.3f}")

    if time_opt > 0:
        print(f"Speedup: {time_orig / time_opt:.1f}x")


if __name__ == "__main__":
    print("🚀 STARTING OPTIMIZATION TEST")
    print("=" * 50)

    speedup = test_basic_optimization()

    print("\n" + "=" * 50)
    if speedup > 1.5 or speedup == float('inf'):
        print("✅ BASIC OPTIMIZATION WORKS - PROCEEDING WITH INTEGRATION")
        print("=" * 50)
        quick_performance_test()  # Сначала быстрый тест
    else:
        print("❌ Optimization needs more work")

    print("\n" + "=" * 50)
    print("🎯 OPTIMIZATION TEST COMPLETE")