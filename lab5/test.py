import pytest
from calcylator import Calculator  # Подразумеваем, что ваш код калькулятора находится в файле calculator.py

@pytest.fixture
def calculator():
    return Calculator()


def test_add(calculator):
    assert calculator.add(1, 2) == 3
    assert calculator.add(-1, 1) == 0
    assert calculator.add(0, 0) == 0