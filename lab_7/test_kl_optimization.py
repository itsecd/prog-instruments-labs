import time
import numpy as np
from IB import kl
from optimized_IB import fast_kl, apply_kl_optimizations_directly


def test_kl_optimization():
    """Тестируем оптимизацию KL дивергенции"""
    print("=== Testing KL Optimization ===")

    # Тестовые распределения
    p = np.random.rand(1000)
    q = np.random.rand(1000)
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Оригинальная версия
    print("Testing ORIGINAL KL...")
    start = time.time()
    for _ in range(100):
        result_orig = kl(p, q)
    time_orig = time.time() - start

    # Оптимизированная версия
    print("Testing OPTIMIZED KL...")
    apply_kl_optimizations_directly()

    start = time.time()
    for _ in range(100):
        result_opt = kl(p, q)  # Теперь использует оптимизированную версию
    time_opt = time.time() - start

    print(f"Original KL: {time_orig:.4f}s")
    print(f"Optimized KL: {time_opt:.4f}s")

    if time_opt > 0:
        speedup = time_orig / time_opt
        print(f"Speedup: {speedup:.1f}x")
    else:
        print("Speedup: Very significant")

    print(f"Result difference: {abs(result_orig - result_opt):.10f}")


def test_kl_correctness():
    """Проверяем корректность оптимизированной KL"""
    print("\n=== Testing KL Correctness ===")

    # Простые тестовые случаи
    test_cases = [
        (np.array([0.5, 0.5]), np.array([0.5, 0.5])),  # Равные распределения
        (np.array([1.0, 0.0]), np.array([0.5, 0.5])),  # Нулевые вероятности
        (np.array([0.8, 0.2]), np.array([0.5, 0.5])),  # Разные распределения
    ]

    apply_kl_optimizations_directly()

    for i, (p, q) in enumerate(test_cases):
        result = kl(p, q)
        print(f"Test {i + 1}: KL({p}, {q}) = {result:.6f}")


if __name__ == "__main__":
    test_kl_optimization()
    test_kl_correctness()