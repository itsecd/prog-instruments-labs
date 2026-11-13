# -*- coding: utf-8 -*-
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
# Создаем форматтер (время - имя логгера - уровень - сообщение)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Обработчик для вывода в файл (все уровни от DEBUG)
file_handler = logging.FileHandler("lab5_execution.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Обработчик для вывода в консоль (только INFO и выше)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Настраиваем корневой логгер
logger = logging.getLogger("NewtonSolver")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- ЗАДАНИЕ 1: Решение системы методом Ньютона (Вариант 8) ---
def F(x):
    return np.array([
        x[0]**2 + 2*x[1]**2 - 3,
        x[0]**2 - x[1]**2 - 1
    ])

def J(x):
    return np.array([
        [2*x[0], 4*x[1]],
        [2*x[0], -2*x[1]]
    ])

def newton_system(F, J, x0, eps=1e-8, max_iter=100):
    logger.info("Запуск метода Ньютона с начальным приближением: %s", x0)
    x = x0.astype(float).copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        Fx = F(x)
        norm_Fx = np.linalg.norm(Fx)
        
        logger.debug("Итерация %d: x = %s, ||F(x)|| = %.2e", i, x, norm_Fx)
        
        if norm_Fx < eps:
            logger.info("Сходимость достигнута на итерации %d (по невязке).", i)
            return x, i, history
            
        Jx = J(x)
        det_Jx = np.linalg.det(Jx)
        
        if abs(det_Jx) < 1e-15:
            logger.error("Матрица Якоби вырождена на итерации %d. Определитель: %.2e", i, det_Jx)
            raise ValueError("Матрица Якоби вырождена")
            
        delta = np.linalg.solve(Jx, -Fx)
        x_new = x + delta
        history.append(x_new.copy())
        
        if np.linalg.norm(x_new - x) < eps:
            logger.info("Сходимость достигнута на итерации %d (по шагу).", i + 1)
            return x_new, i + 1, history
            
        x = x_new
        
    logger.error("Метод не сошелся за %d итераций.", max_iter)
    raise RuntimeError("Метод не сошелся")

# --- ОСНОВНОЙ БЛОК ВЫПОЛНЕНИЯ ---
logger.info("=== СТАРТ ПРОГРАММЫ: Лабораторная работа №5 ===")
x0 = np.array([1.5, 0.5])
logger.info("Начальные параметры: x0 = %s, точность eps = 1e-8", x0)

try:
    root, iters, history = newton_system(F, J, x0)
    logger.info("Найденный корень: x = [%.8f, %.8f]", root[0], root[1])
    logger.info("Число итераций: %d", iters)
    logger.info("Невязка ||F(x)||: %.2e", np.linalg.norm(F(root)))
except Exception as e:
    logger.critical("Критическая ошибка при решении системы: %s", e)

# 2. Проверка точности через scipy
try:
    from scipy.optimize import fsolve
    logger.info("Запуск проверки через scipy.optimize.fsolve...")
    root_scipy = fsolve(F, x0)
    diff = np.linalg.norm(root - root_scipy)
    logger.info("Решение scipy: %s. Разница с нашим методом: %.2e", root_scipy, diff)
except ImportError:
    logger.warning("Библиотека scipy не установлена. Проверка пропущена.")

# 3. Исследование сходимости для разных начальных приближений
logger.info("Запуск исследования сходимости для разных начальных точек.")
starts = [(1.5, 0.5), (1.0, 1.0), (2.0, 1.0), (-1.5, -0.5), (0.5, 0.5)]

for x0_val, y0_val in starts:
    try:
        x0_vec = np.array([x0_val, y0_val])
        r_tmp, it_tmp, _ = newton_system(F, J, x0_vec)
        logger.info("Точка (%s, %s): сошлось за %d итераций.", x0_val, y0_val, it_tmp)
    except Exception as e:
        logger.warning("Точка (%s, %s): НЕ сошлось. Причина: %s", x0_val, y0_val, e)

# 4. Визуализация
logger.info("Построение и сохранение графика...")
x_vals = np.linspace(-2.5, 2.5, 400)
y_vals = np.linspace(-2.5, 2.5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
F1 = X**2 + 2*Y**2 - 3
F2 = X**2 - Y**2 - 1

plt.figure(figsize=(8, 8))
plt.contour(X, Y, F1, levels=[0], colors='blue', linewidths=2)
plt.contour(X, Y, F2, levels=[0], colors='green', linewidths=2)
hist_arr = np.array(history)
plt.plot(hist_arr[:, 0], hist_arr[:, 1], 'ro-', label='Итерации')
plt.plot(root[0], root[1], 'ks', markersize=10, label='Корень')
plt.title('Метод Ньютона для системы (Вариант 8)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')

output_filename = 'lab5_newton_system.png'
plt.savefig(output_filename, dpi=300)
logger.info("График успешно сохранен в файл: %s", output_filename)
plt.close()
logger.info("=== ЗАВЕРШЕНИЕ ПРОГРАММЫ ===")