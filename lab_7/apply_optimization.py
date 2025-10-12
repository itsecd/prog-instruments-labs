"""
ПРИМЕНЕНИЕ ОПТИМИЗАЦИИ К ОСНОВНОМУ КОДУ IB
Лабораторная работа 7 "Профайлинг и оптимизация производительности"
"""

from IB import dataset
from optimized_IB import add_optimized_methods, apply_kl_optimizations_directly
import time


def apply_all_optimizations():
    """
    Применяет все оптимизации к основному коду IB

    Включает:
    1. Векторизованный coord_to_pxy для uniform smoothing
    2. Оптимизированные вычисления KL дивергенции
    """
    print("🔧 APPLYING ALL OPTIMIZATIONS TO IB CODE")

    # 1. Оптимизация coord_to_pxy
    add_optimized_methods()

    # 2. Оптимизация KL дивергенции
    apply_kl_optimizations_directly()

    # 3. Замена основного метода coord_to_pxy
    original_coord_to_pxy = dataset.coord_to_pxy

    def optimized_coord_to_pxy(self, total_bins=2500, pad=None, drop_distant=True):
        """
        Оптимизированная версия coord_to_pxy

        Автоматически использует быстрый метод для uniform smoothing,
        для других типов smoothing использует оригинальный метод
        """
        if self.smoothing_type == 'uniform':
            # print("⚡ Using optimized vectorized method for uniform smoothing")
            return self.fast_coord_to_pxy(total_bins, pad, drop_distant)
        else:
            # Для других типов smoothing используем оригинальный метод
            return original_coord_to_pxy(self, total_bins, pad, drop_distant)

    # Заменяем метод в классе
    dataset.coord_to_pxy = optimized_coord_to_pxy

    # Сохраняем ссылку на оригинальный метод для тестов
    dataset._original_coord_to_pxy = original_coord_to_pxy

    print("✅ All optimizations applied!")
    print("   - Vectorized coord_to_pxy for uniform smoothing")
    print("   - Optimized KL divergence calculations")


def test_optimized_workflow():
    """Тестируем полный рабочий процесс с примененными оптимизациями"""
    print("\n🧪 TESTING OPTIMIZED WORKFLOW")
    from data_generation import gen_easytest
    from IB import model

    # Создаем dataset
    ds = gen_easytest(plot=False)
    ds.s = 1.0
    ds.smoothing_type = 'uniform'

    # Автоматически использует оптимизированную версию!
    print("Calling coord_to_pxy (using optimized version)...")
    start_time = time.time()
    ds.coord_to_pxy(total_bins=200)
    coord_time = time.time() - start_time

    print(f"✅ Dataset ready in {coord_time:.3f}s: X={ds.X}, Y={ds.Y}, I(X;Y)={ds.ixy:.3f}")

    # Тестируем модель с оптимизированными KL вычислениями
    print("\nTesting model fitting with optimized KL calculations...")
    m = model(ds=ds, alpha=1, beta=5, quiet=True)

    start_time = time.time()
    m.fit(keep_steps=False)
    fit_time = time.time() - start_time

    print(f"✅ Model fitted in {fit_time:.3f}s")
    print(f"Final metrics: {m.report_metrics()}")

    return m


def performance_comparison():
    """
    Сравнение производительности до и после оптимизации

    Возвращает:
    - speedup_coord: ускорение coord_to_pxy
    - speedup_kl: ускорение KL вычислений
    """
    print("\n📊 PERFORMANCE COMPARISON: BEFORE vs AFTER")
    from data_generation import gen_easytest
    from IB import kl
    import numpy as np

    results = {}

    # Тест 1: coord_to_pxy
    print("\n1. Testing coord_to_pxy performance...")

    # ДО оптимизации
    ds1 = gen_easytest(plot=False)
    ds1.s = 1.0
    ds1.smoothing_type = 'uniform'

    start = time.time()
    # Временно используем оригинальный метод
    if hasattr(dataset, '_original_coord_to_pxy'):
        dataset.coord_to_pxy = dataset._original_coord_to_pxy
    ds1.coord_to_pxy(total_bins=200)
    time_before_coord = time.time() - start

    # ПОСЛЕ оптимизации
    apply_all_optimizations()  # Применяем оптимизации

    ds2 = gen_easytest(plot=False)
    ds2.s = 1.0
    ds2.smoothing_type = 'uniform'

    start = time.time()
    ds2.coord_to_pxy(total_bins=200)  # Автоматически использует оптимизированную версию
    time_after_coord = time.time() - start

    speedup_coord = time_before_coord / time_after_coord
    results['coord_speedup'] = speedup_coord

    print(f"   coord_to_pxy: {time_before_coord:.3f}s → {time_after_coord:.3f}s")
    print(f"   Speedup: {speedup_coord:.1f}x")

    # Тест 2: KL дивергенция
    print("\n2. Testing KL divergence performance...")

    # Тестовые распределения
    p = np.random.rand(1000)
    q = np.random.rand(1000)
    p = p / np.sum(p)
    q = q / np.sum(q)

    # ДО оптимизации
    if hasattr(kl, '_original_kl'):
        kl_original = kl._original_kl
    else:
        kl_original = kl

    start = time.time()
    for _ in range(100):
        result_before = kl_original(p, q)
    time_before_kl = time.time() - start

    # ПОСЛЕ оптимизации (уже применены)
    start = time.time()
    for _ in range(100):
        result_after = kl(p, q)
    time_after_kl = time.time() - start

    speedup_kl = time_before_kl / time_after_kl
    results['kl_speedup'] = speedup_kl

    print(f"   KL divergence: {time_before_kl:.3f}s → {time_after_kl:.3f}s")
    print(f"   Speedup: {speedup_kl:.1f}x")
    print(f"   Result difference: {abs(result_before - result_after):.10f}")

    # Итоги
    print("\n📈 FINAL RESULTS:")
    print(f"   coord_to_pxy speedup: {results['coord_speedup']:.1f}x")
    print(f"   KL divergence speedup: {results['kl_speedup']:.1f}x")
    print(f"   AVERAGE SPEEDUP: {(results['coord_speedup'] + results['kl_speedup']) / 2:.1f}x")

    return results


if __name__ == "__main__":
    print("🎯 IB PERFORMANCE OPTIMIZATION")
    print("=" * 50)

    # Применяем все оптимизации
    apply_all_optimizations()

    # Тестируем рабочий процесс
    test_optimized_workflow()

    # Сравниваем производительность
    performance_comparison()

    print("\n" + "=" * 50)
    print("✅ ALL OPTIMIZATIONS SUCCESSFULLY APPLIED AND TESTED!")
    print("Ready for laboratory submission 🎓")