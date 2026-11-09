import pytest
from src.calc import Calculator


@pytest.fixture
def calc() -> Calculator:
    return Calculator()


def test_add(calc: Calculator) -> None:
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0


def test_subtract(calc: Calculator) -> None:
    assert calc.subtract(5, 3) == 2


def test_multiply(calc: Calculator) -> None:
    assert calc.multiply(4, 3) == 12


def test_divide(calc: Calculator) -> None:
    assert calc.divide(10, 2) == 5.0


def test_divide_by_zero(calc: Calculator) -> None:
    with pytest.raises(ValueError, match="Cannot divide by zero."):
        calc.divide(10, 0)
