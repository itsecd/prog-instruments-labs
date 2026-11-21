import numpy as np
import math

# Вариант 8: f(x) = e^(-x^2)
def f(x):
    return np.exp(-x**2)

def df_analytical(x):
    return -2.0 * x * np.exp(-x**2)

# --- ЗАДАНИЕ 1: Численное дифференцирование ---
def left_diff(f, x, h): 
    return (f(x) - f(x - h)) / h

def right_diff(f, x, h): 
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h): 
    return (f(x + h) - f(x - h)) / (2 * h)

# --- ЗАДАНИЕ 3: Численное интегрирование ---
def rectangle_left(f, a, b, n):
    h = (b - a) / n
    return h * np.sum(f(np.linspace(a, b - h, n)))

def rectangle_right(f, a, b, n):
    h = (b - a) / n
    return h * np.sum(f(np.linspace(a + h, b, n)))

def rectangle_mid(f, a, b, n):
    h = (b - a) / n
    return h * np.sum(f(np.linspace(a + h/2, b - h/2, n)))

def trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return h * (0.5 * f(a) + np.sum(f(x[1:-1])) + 0.5 * f(b))

def simpson(f, a, b, n):
    if n % 2 == 1: 
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    result = f(a) + f(b)
    for i in range(1, n):
        result += (4 if i % 2 == 1 else 2) * f(x[i])
    return h / 3 * result