def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b


def main():
    # Ошибка типов: передаем строку туда, где ожидается число
    result = add(10, 20)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
