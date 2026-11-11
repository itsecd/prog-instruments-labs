"""CLI-интерфейс калькулятора."""

import argparse

from calc.operations import add, divide, multiply, subtract

OPERATIONS = {
    "add": add,
    "sub": subtract,
    "mul": multiply,
    "div": divide,
}


def main() -> None:
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(description="Простой калькулятор")
    parser.add_argument("op", choices=OPERATIONS, help="Операция")
    parser.add_argument("a", type=float, help="Первое число")
    parser.add_argument("b", type=float, help="Второе число")
    args = parser.parse_args()

    func = OPERATIONS[args.op]
    try:
        result = func(args.a, args.b)
    except ZeroDivisionError as exc:
        print(f"Ошибка: {exc}")
        raise SystemExit(1) from exc
    print(result)


if __name__ == "__main__":
    main()