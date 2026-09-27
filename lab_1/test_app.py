import math
import numpy as np
import pytest
from lab_1 import f_32, f_64, machine_eps, my_function


def test_functions_at_zero():
    assert f_64(0.0) == np.float64(1.0)
    assert f_32(0.0) == np.float32(1.0)


def test_machine_epsilon():
    eps32 = machine_eps(np.float32)
    eps64 = machine_eps(np.float64)
    assert eps32 > 0
    assert eps64 > 0
    assert eps64 < eps32


def test_my_function_convergence():
    x = 0.5
    eps = 1e-6
    approx, iters = my_function(x, eps)
    exact = 1.0 / math.sqrt(1.0 - x)
    assert abs(approx - exact) < 1e-4
    assert iters > 0


def test_my_function_divergence():
    with pytest.raises(ValueError):
        my_function(1.2, 1e-6)