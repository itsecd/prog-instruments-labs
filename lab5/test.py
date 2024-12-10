import pytest
from calcylator import Calculator  # Подразумеваем, что ваш код калькулятора находится в файле calculator.py

@pytest.fixture
def calculator():
    return Calculator()


def test_add(calculator):
    assert calculator.add(1, 2) == 3
    assert calculator.add(-1, 1) == 0
    assert calculator.add(0, 0) == 0


def test_subtract(calculator):
    assert calculator.subtract(5, 3) == 2
    assert calculator.subtract(10, 10) == 0
    assert calculator.subtract(4, 7) == -3


def test_multiply(calculator):
    assert calculator.multiply(3, 4) == 12
    assert calculator.multiply(-2, 5) == -10
    assert calculator.multiply(0, 10) == 0


def test_divide(calculator):
    assert calculator.divide(10, 2) == 5
    assert calculator.divide(-6, 2) == -3
    with pytest.raises(ValueError, match="Cannot divide by zero."):
        calculator.divide(5, 0)