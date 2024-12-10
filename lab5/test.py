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

@pytest.mark.parametrize("a, b, expected", [
    (1, 1, 2),
    (0, 0, 0),
    (-1, -1, -2),
    (100, 200, 300)
])
def test_parametrized_add(calculator, a, b, expected):
    assert calculator.add(a, b) == expected


@pytest.mark.parametrize("num, expected", [
    (4, 2),
    (0, 0),
    (1, 1),
    (9, 3),
    (100, 10),
])
def test_square_root(calculator, num, expected):
    assert calculator.square_root(num) == expected


def test_square_root_negative(calculator):
    with pytest.raises(ValueError, match="Cannot take the square root of a negative number."):
        calculator.square_root(-1)


def test_square(calculator):
    assert calculator.square(3) == 9
    assert calculator.square(-3) == 9
    assert calculator.square(0) == 0