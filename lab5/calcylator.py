import math

class Calculator:
    def add(self, a: float, b: float) -> float:
        return a + b

    def subtract(self, a: float, b: float) -> float:
        return a - b

    def multiply(self, a: float, b: float) -> float:
        return a * b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    def square(self, a: float) -> float:
        return a ** 2

    def square_root(self, a: float) -> float:
        if a < 0:
            raise ValueError("Cannot take the square root of a negative number.")
        return math.sqrt(a)


def main():
    calc = Calculator()

    print("Выберите функцию:")
    print("1. Сложение")
    print("2. Вычитание")
    print("3. Умножение")
    print("4. Деление")
    print("5. Возведение в квадрат")
    print("6. Извлечение квадратного корня")

    choice = input("Введите номер функции (1-6): ")

    if choice in ['1', '2', '3', '4']:
        a = float(input("Введите первое число: "))
        b = float(input("Введите второе число: "))

        if choice == '1':
            print(f"Результат: {calc.add(a, b)}")
        elif choice == '2':
            print(f"Результат: {calc.subtract(a, b)}")
        elif choice == '3':
            print(f"Результат: {calc.multiply(a, b)}")
        elif choice == '4':
            try:
                print(f"Результат: {calc.divide(a, b)}")
            except ValueError as e:
                print(e)

    elif choice == '5':
        a = float(input("Введите число: "))
        print(f"Результат: {calc.square(a)}")

    elif choice == '6':
        a = float(input("Введите число: "))
        try:
            print(f"Результат: {calc.square_root(a)}")
        except ValueError as e:
            print(e)

    else:
        print("Неверный выбор.")


if __name__ == "__main__":
    main()