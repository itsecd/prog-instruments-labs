import cProfile
import pstats
import io
import numpy as np
from IB import *
from data_generation import *


def profile_coord_to_pxy():
    """Профилируем самую тяжелую часть - преобразование координат в pxy"""
    print("=== Profiling coord_to_pxy ===")
    ds = gen_easytest(plot=False)
    ds.s = 1.

    pr = cProfile.Profile()
    pr.enable()
    ds.coord_to_pxy()  # возможный bottleneck
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())


def profile_small_dataset():
    """Профилируем на уменьшенном датасете для скорости"""
    print("=== Profiling with small dataset ===")
    # Создаем очень маленький датасет для быстрого профайлинга
    ds = dataset(
        coord=np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),  # всего 5 точек
        labels=np.array([0, 0, 1, 1, 1]),
        name="tiny_test"
    )
    ds.s = 0.5
    ds.smoothing_type = 'uniform'

    pr = cProfile.Profile()
    pr.enable()
    ds.coord_to_pxy(total_bins=100)  # Меньше бинов для скорости
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)
    print(s.getvalue())


def profile_kl_calculations():
    """Профилируем конкретно KL-расчеты"""
    print("=== Profiling KL calculations ===")
    # Создаем тестовые распределения
    p = np.random.rand(100)
    q = np.random.rand(100)
    p = p / np.sum(p)
    q = q / np.sum(q)

    pr = cProfile.Profile()
    pr.enable()
    # Многократно вызываем KL для профилирования
    for _ in range(1000):
        result = kl(p, q)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('time')
    ps.print_stats(10)
    print(s.getvalue())


if __name__ == "__main__":
    print("Starting IB profiling...")

    # Начнем с самого маленького теста
    #profile_small_dataset()

    # Потом можно раскомментировать другие
    profile_coord_to_pxy()
    profile_kl_calculations()
