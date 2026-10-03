"""Тесты для lab1.py."""

import math

import numpy as np

from lab1 import func, machine_epsilon, my_func


def test_func_near_zero() -> None:
    """При x, близком к нулю, функция возвращает предел 1."""
    assert func(np.float64(0.0)) == 1.0


def test_func_regular_value() -> None:
    """Обычное значение совпадает с math.exp."""
    expected = (math.exp(1.0) - 1.0) / 1.0
    assert math.isclose(float(func(np.float64(1.0))), expected, rel_tol=1e-12)


def test_machine_epsilon() -> None:
    """Машинный эпсилон float64 и float32."""
    assert math.isclose(machine_epsilon(np.float64), 2.22e-16, rel_tol=1e-2)
    assert math.isclose(machine_epsilon(np.float32), 1.19e-7, rel_tol=1e-2)


def test_my_func_converges() -> None:
    """Ряд Тейлора сходится к точному значению."""
    approx, terms = my_func(1.0, 1e-10)
    assert math.isclose(approx, math.exp(1.0) - 1.0, rel_tol=1e-8)
    assert terms > 0


def test_my_func_zero() -> None:
    """При x = 0 возвращается предел 1."""
    assert my_func(0.0) == (1.0, 0)
