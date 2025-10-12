"""
ПРИМЕНЕНИЕ ОПТИМИЗАЦИИ К ОСНОВНОМУ КОДУ IB
"""

from IB import dataset
from optimized_IB import add_optimized_methods
import time


def apply_optimizations():
    """Применяет все оптимизации к основному коду"""
    print("🔧 APPLYING OPTIMIZATIONS TO IB CODE")

    # Добавляем оптимизированные методы
    add_optimized_methods()

    # Сохраняем оригинальный метод
    original_coord_to_pxy = dataset.coord_to_pxy

    def optimized_coord_to_pxy(self, total_bins=2500, pad=None, drop_distant=True):
        """Оптимизированная версия, которая использует быстрый метод для uniform smoothing"""
        if self.smoothing_type == 'uniform':
            print("⚡ Using optimized vectorized method for uniform smoothing")
            return self.fast_coord_to_pxy(total_bins, pad, drop_distant)
        else:
            # Для других типов smoothing используем оригинальный метод
            print("Using original method for non-uniform smoothing")
            return original_coord_to_pxy(self, total_bins, pad, drop_distant)

    # Заменяем метод в классе
    dataset.coord_to_pxy = optimized_coord_to_pxy
    print("✅ Optimizations applied! coord_to_pxy now uses vectorized version for uniform smoothing")


def test_optimized_workflow():
    """Тестируем полный рабочий процесс с оптимизациями"""
    print("\n🧪 TESTING OPTIMIZED WORKFLOW")
    from data_generation import gen_easytest
    from IB import model

    # Создаем dataset
    ds = gen_easytest(plot=False)
    ds.s = 1.0
    ds.smoothing_type = 'uniform'

    # Это теперь автоматически использует оптимизированную версию!
    print("Calling coord_to_pxy (should use optimized version automatically)...")
    start_time = time.time()
    ds.coord_to_pxy(total_bins=200)
    coord_time = time.time() - start_time

    print(f"✅ Dataset ready in {coord_time:.3f}s: X={ds.X}, Y={ds.Y}, I(X;Y)={ds.ixy:.3f}")

    # Тестируем модель
    print("\nTesting model fitting with optimized dataset...")
    m = model(ds=ds, alpha=1, beta=5, quiet=True)

    start_time = time.time()
    m.fit(keep_steps=False)
    fit_time = time.time() - start_time

    print(f"✅ Model fitted in {fit_time:.3f}s")
    print(f"Final: {m.report_metrics()}")

    return m


def performance_comparison():
    """Сравнение производительности до и после оптимизации"""
    print("\n📊 PERFORMANCE COMPARISON")
    from data_generation import gen_easytest

    # Тест до оптимизации
    print("BEFORE optimization:")
    ds1 = gen_easytest(plot=False)
    ds1.s = 1.0
    ds1.smoothing_type = 'uniform'

    start = time.time()
    # Временно используем оригинальный метод
    original_method = dataset.coord_to_pxy
    dataset.coord_to_pxy = dataset.__dict__.get('_original_coord_to_pxy', original_method)
    ds1.coord_to_pxy(total_bins=200)
    time_before = time.time() - start

    # Тест после оптимизации
    print("AFTER optimization:")
    apply_optimizations()  # Применяем оптимизации

    ds2 = gen_easytest(plot=False)
    ds2.s = 1.0
    ds2.smoothing_type = 'uniform'

    start = time.time()
    ds2.coord_to_pxy(total_bins=200)  # Автоматически использует оптимизированную версию
    time_after = time.time() - start

    print(f"\n📈 RESULTS:")
    print(f"Before: {time_before:.3f}s")
    print(f"After:  {time_after:.3f}s")
    print(f"Speedup: {time_before / time_after:.1f}x")

    # Восстанавливаем оригинальный метод для тестов
    dataset._original_coord_to_pxy = original_method


if __name__ == "__main__":
    print("🎯 APPLYING IB OPTIMIZATIONS")
    print("=" * 50)

    apply_optimizations()
    test_optimized_workflow()
    performance_comparison()

    print("\n" + "=" * 50)
    print("✅ ALL OPTIMIZATIONS SUCCESSFULLY APPLIED!")