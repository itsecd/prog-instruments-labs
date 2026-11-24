"""
ЛАБОРАТОРНАЯ РАБОТА: Проверка статистических гипотез о нормальности, расчет мощности и критерии согласия

ОБЩАЯ ИНСТРУКЦИЯ ДЛЯ СТУДЕНТОВ:
1. Запустите этот скрипт в вашей среде разработки (PyCharm, VS Code, IDLE).
2. Графики будут открываться в отдельных всплывающих окнах. Чтобы программа шла дальше, просто закрывайте окно с графиком.
3. Измените значение переменной MY_VARIANT на свой номер по списку.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm

# ==========================================
# ЧАСТЬ 2. ИНИЦИАЛИЗАЦИЯ И ВЫБОР ВАРИАНТА
# ==========================================
# УКАЖИТЕ СВОЙ НОМЕР ВАРИАНТА НИЖЕ:
MY_VARIANT = 2  

print(f"--- Активирован вариант №{MY_VARIANT} ---")

# ==========================================
# ЗАДАНИЕ 1. Визуальное сравнение плотностей
# ==========================================
print("[Задание 1] Расчет теоретических плотностей...")
x = np.linspace(-4, 4, 1000)
pdf_norm = stats.norm.pdf(x, loc=0, scale=1)
pdf_t_20 = stats.t.pdf(x, df=20)
pdf_t_2  = stats.t.pdf(x, df=2)

plt.figure(figsize=(10, 5))
plt.plot(x, pdf_norm, label='Нормальное N(0, 1)', color='black', linewidth=2.5)
plt.plot(x, pdf_t_20, label='Стьюдента (df = 20)', color='red', linestyle='--', linewidth=2)
plt.plot(x, pdf_t_2, label='Стьюдента (df = 2)', color='blue', linestyle=':', alpha=0.6)
plt.title('Сравнение теоретических плотностей распределений')
plt.xlabel('Значение (X)')
plt.ylabel('Плотность вероятности')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
print("Окно с графиком открыто. Закройте его, чтобы продолжить расчеты...")
plt.show() 

max_diff = np.max(np.abs(pdf_norm - pdf_t_20))
print(f"Максимальная разница между N(0,1) и t(df=20): {max_diff:.5f}")

# ==========================================
# ЗАДАНИЕ 2. Парадокс больших выборок
# ==========================================
print("[Задание 2] Запуск тестов на нормальность для выборок разного объема...")
np.random.seed(MY_VARIANT)
df_student = 20

samples = {
    'sample_30': stats.t.rvs(df=df_student, size=30),
    'sample_300': stats.t.rvs(df=df_student, size=300),
    'sample_5000': stats.t.rvs(df=df_student, size=5000)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
results = []

for i, (name, data) in enumerate(samples.items()):
    N = len(data)
    stat_sw, p_sw = stats.shapiro(data)
    stat_dag, p_dag = stats.normaltest(data)
    с = "По Шапиро-Уилок H_0 принимаем"
    if (p_sw < 0.05):
        с = "По Шапиро-Уилок H_0 отклоняем"
    s = "По Д'Агостино H_0 принимаем"
    if (p_dag < 0.05):
        s = "По Д'Агостино H_0 отклоняем"
        
    results.append({
        'Выборка': name,
        'Размер (N)': N,
        'p-value (Шапиро-Уилок)': round(p_sw, 5),
        'p-value (Д\'Агостино)': round(p_dag, 5)
    })
    
    sns.histplot(data, kde=True, stat="density", color="lightgreen", ax=axes[i], alpha=0.6)
    x_plot = np.linspace(data.min(), data.max(), 100)
    axes[i].plot(x_plot, stats.norm.pdf(x_plot, loc=data.mean(), scale=data.std()), color='red', linestyle='--', linewidth=2, label='Теор. нормальное')
    axes[i].set_title(f'{name} (N = {N})\np-val SW: {p_sw:.8f}\np-val Dag: {p_dag:.8f}')
    axes[i].legend()

plt.tight_layout()
print("Окно с сеткой графиков открыто. Закройте его...")
plt.show()

df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
print("\n")
print("\n")

# ==========================================
# ЗАДАНИЕ 3. Мощность критерия (Монте-Карло)
# ==========================================
# Задача: Эмпирически оценить мощность критерия Шапиро–Уилка для объемов выборок 𝑛=30, 𝑛=300 и 𝑛=5000 при генерации данных из распределения Стьюдента (𝑑𝑓=20).

# Инструкция:

#     Допишите код симуляции Монте-Карло для каждого объема выборки.
#     Проведите 10000 симуляций для каждого 𝑛, фиксируя долю случаев, когда полученный 𝑝-value≤0.05.

n_simulations = 10000
sample_sizes = [30, 300, 5000]
alpha = 0.05
 
power_results = {}
 
for n in sample_sizes:
    rejections = 0
    for _ in tqdm(range(n_simulations), desc=f"n={n}"):
        sample = stats.t.rvs(df=df_student, size=n)
        _, p_value = stats.shapiro(sample)
        if p_value <= alpha:
            rejections += 1
            
    power = rejections / n_simulations
    power_results[n] = power
 
print("\n--- Результаты: мощность критерия Шапиро-Уилка ---")
for n, power in power_results.items():
    print(f"n = {n}: мощность = {power:.4f} ({power * 100:.2f}%)")