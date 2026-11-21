import numpy as np
import time
import cProfile
import pstats
import io
import math

# Функция из вашей лабы 6
def f(x):
    return np.exp(-x**2)

# --- МЕДЛЕННАЯ ВЕРСИЯ (с циклом, как в lab6.py) ---
def simpson_slow(f, a, b, n):
    if n % 2 == 1: 
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    result = f(a) + f(b)
    for i in range(1, n):
        result += (4 if i % 2 == 1 else 2) * f(x[i])
    return h / 3 * result

# --- БЫСТРАЯ ВЕРСИЯ (оптимизированная, векторизованная) ---
def simpson_fast(f, a, b, n):
    if n % 2 == 1: 
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    
    # Создаем массив коэффициентов (1, 4, 2, 4, 2, ..., 4, 1)
    coeffs = np.ones(n + 1)
    coeffs[1:-1:2] = 4  # нечетные индексы
    coeffs[2:-1:2] = 2  # четные индексы
    
    # Умножаем значения функции на коэффициенты и суммируем
    result = np.sum(coeffs * f(x))
    return h / 3 * result

# --- ДЕМОНСТРАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ ---
a, b = 0.0, 1.0
n = 1000000 # Большое число разбиений, чтобы разница во времени была заметна

print("Запуск медленной версии...")
start = time.time()
res_slow = simpson_slow(f, a, b, n)
time_slow = time.time() - start
print(f"Результат: {res_slow:.10f}, Время: {time_slow:.4f} сек")

print("\nЗапуск быстрой версии...")
start = time.time()
res_fast = simpson_fast(f, a, b, n)
time_fast = time.time() - start
print(f"Результат: {res_fast:.10f}, Время: {time_fast:.4f} сек")

print(f"\nУскорение: в {time_slow / time_fast:.1f} раз!")

# --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ПРОФИЛИРОВАНИЯ ---
print("\nСбор профиля медленной версии (cProfile)...")
pr = cProfile.Profile()
pr.enable()
simpson_slow(f, a, b, n)
pr.disable()

# Сохраняем отчет в файл, как требует преподаватель
with open("profiling_report.txt", "w", encoding="utf-8") as f_out:
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10) # Выводим топ-10 самых затратных операций
    f_out.write(s.getvalue())

print("Отчет профилирования сохранен в файл profiling_report.txt")