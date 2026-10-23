class Calculator:

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(result)
        return result

    def divide(self, a, b):
        if b == 0:
            print("Ошибка: деление на ноль")
            return None
        result = a / b
        self.history.append(result)
        return result

    def multiply(self, a, b):
        result = a * b
        self.history.append(result)
        return result

    def get_history(self):
        return self.history.copy()

    def clear_history(self):
        self.history.clear()


def format_result(value):
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