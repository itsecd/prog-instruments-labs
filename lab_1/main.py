from typing import Union


def add(a: int, b: int) -> int:
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b


def multiply(a: int, b: int) -> int:
    return a * b


def divide(a: int, b: int) -> Union[float, str]:
    if b == 0:
        return "Error: Division by zero"
    return a / b


def main() -> None:
    result = add(10, 20)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
