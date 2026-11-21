import pytest
import math
import numpy as np
from unittest.mock import Mock
from math_methods import (
    f, df_analytical, left_diff, central_diff, 
    rectangle_left, trapezoidal, simpson
)

# --- ПРОСТЫЕ ТЕСТЫ ---

def test_function_f():
    """Тест 1: Проверка самой функции f(x) = e^(-x^2)"""
    assert f(0) == 1.0
    assert pytest.approx(f(1), rel=1e-5) == math.exp(-1)

def test_derivative_analytical():
    """Тест 2: Проверка аналитической производной"""
    assert df_analytical(0) == 0.0
    assert pytest.approx(df_analytical(1), rel=1e-5) == -2 * math.exp(-1)

def test_rectangle_left_simple():
    """Тест 3: Базовая проверка метода левых прямоугольников"""
    def linear_func(x): return x
    # Интеграл от f(x)=x на [0,1] равен 0.5. 
    # Левые прямоугольники (n=10) дадут 0.45
    assert pytest.approx(rectangle_left(linear_func, 0, 1, 10)) == 0.45

def test_simpson_simple():
    """Тест 4: Проверка метода Симпсона на линейной функции"""
    # Интеграл от f(x) = x на [0, 1] равен 0.5
    def linear_func(x): return x
    assert pytest.approx(simpson(linear_func, 0, 1, 10)) == 0.5

# --- НЕПРОСТЫЕ ТЕСТЫ (Параметризация) ---

@pytest.mark.parametrize("h, expected_diff", [
    (1e-5, 0.0), 
    (0.01, 0.0)
])
def test_central_diff_zero(h, expected_diff):
    """Тест 5 (Параметризованный): Проверка центральной разности в точке экстремума"""
    # Производная e^(-x^2) в точке 0 равна 0
    assert pytest.approx(central_diff(f, 0.0, h), abs=1e-6) == expected_diff

@pytest.mark.parametrize("n, a, b, expected", [
    (10, 0, 1, 0.746824),
    (100, 0, 1, 0.746824)
])
def test_trapezoidal_accuracy(n, a, b, expected):
    """Тест 6 (Параметризованный): Проверка точности метода трапеций"""
    assert pytest.approx(trapezoidal(f, a, b, n), abs=1e-3) == expected

# --- СЛОЖНЫЙ ТЕСТ (Использование mock/stub) ---

def test_rectangle_left_mock():
    """Тест 7: Использование мока для проверки вызовов внутри функции"""
    from unittest.mock import MagicMock
    import numpy as np
    
    # Создаем мок, который умеет работать с массивами и возвращает 1.0
    spy_func = MagicMock(side_effect=lambda x: np.ones_like(x))
    
    result = rectangle_left(spy_func, 0, 1, 5)
    
    # Проверяем, что функция вызывалась ровно 1 раз (так как numpy передает весь массив сразу)
    assert spy_func.call_count == 1

# --- ТЕСТ НА ИСКЛЮЧЕНИЯ (Edge cases) ---

def test_simpson_odd_n():
    """Тест 8: Проверка, что метод Симпсона корректно обрабатывает нечетное n"""
    # Если передать n=5, функция внутри делает n+=1 (то есть 6)
    # Сравниваем результат с n=6
    res_odd = simpson(f, 0, 1, 5)
    res_even = simpson(f, 0, 1, 6)
    assert res_odd == res_even