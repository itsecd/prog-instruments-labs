class Calculator:
    def __init__(self) -> None:
        self.history: list[float] = []

    def add(self, a: float, b: float) -> float:
        result = a + b
        self.history.append(result)
        return result

    def divide(self, a: float, b: float) -> float | None:
        if b == 0:
            print("Ошибка: деление на ноль")
            return None
        result = a / b
        self.history.append(result)
        return result

    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self.history.append(result)
        return result

    def get_history(self) -> list[float]:
        return self.history.copy()

    def clear_history(self) -> None:
        self.history.clear()


def format_result(value: float | None) -> str:
    if value is None:
        return "Результат отсутствует"
    return "Результат: " + str(value)


if __name__ == "__main__":
    calc = Calculator()

    print(calc.add(10, 5))
    res_zero = calc.divide(10, 0)
    print(calc.divide(10, 2))

    res_mult = calc.multiply(3, 4)
    print(res_mult)

    print(format_result(res_mult))
    print(format_result(res_zero))

    print("История операций:", calc.get_history())
