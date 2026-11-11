"""Арифметические операции калькулятора."""


def add(a: float, b: float) -> float:
    """Сложить два числа."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Вычесть второе число из первого."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Умножить два числа."""
    return a * b


def divide(a: float, b: float) -> float:
    """Разделить первое число на второе.

    Raises:
        ZeroDivisionError: если второе число равно нулю.
    """
    if b == 0:
        raise ZeroDivisionError("Деление на ноль невозможно")
    return a / b