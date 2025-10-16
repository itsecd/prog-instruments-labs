import pytest
import builtins
from calculator import*

def test_addition(monkeypatch):
    monkeypatch.setattr(builtins, 'input', lambda _: "2 5")
    assert addition() == 7

def test_subtraction(monkeypatch):
    inputs = iter(["5", "2"])
    monkeypatch.setattr(builtins, 'input', lambda _: next(inputs))
    assert subtraction() == 3

def test_multiplication(monkeypatch):
    monkeypatch.setattr(builtins, 'input', lambda _: "2 5")
    assert multiplication() == 10

def test_exponentiation(monkeypatch):
    inputs = iter(["15", "3"])
    monkeypatch.setattr(builtins, 'input', lambda _: next(inputs))
    assert division() == 5

def test_exponentiation(monkeypatch):
    inputs = iter(["5", "0"])
    monkeypatch.setattr(builtins, 'input', lambda _: next(inputs))
    assert division() == "Invalid entry"

def test_factorial(monkeypatch):
    assert factorial(2) == 2


@pytest.mark.parametrize("input,result",
    [
        ([1, 2, 3, 4], 2.5),
        ([10, 20], 15),
        ([5], 5),
    ]
)
def test_average(monkeypatch, input, result):
    monkeypatch.setattr(builtins, 'input', lambda _: " ".join(map(str, input)))
    assert average() == pytest.approx(result)


@pytest.mark.parametrize("n, expected",
                         [
                             ([5, 2], 10),
                             ([10, 2], 45),
                         ]
                         )
def test_binomial(n, expected):
    assert binomial(n) == expected

def test_complex(monkeypatch):
    inputs = iter(["1", "1 2 3 4"])
    monkeypatch.setattr(builtins, 'input', lambda _: next(inputs))
    result = complex_arithmetic()
    assert result == "4 + i7"

def test_complex_division_by_zero(monkeypatch):
    inputs = iter(["4", "1 2 0 0"])
    monkeypatch.setattr(builtins, 'input', lambda _: next(inputs))
    with pytest.raises(ZeroDivisionError):
        complex_arithmetic()

