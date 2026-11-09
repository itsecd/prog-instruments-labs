from calc import Calculator


def main() -> None:
    calc = Calculator()

    print(f"10 + 5 = {calc.add(10, 5)}")
    print(f"10 - 5 = {calc.subtract(10, 5)}")
    print(f"10 * 5 = {calc.multiply(10, 5)}")
    print(f"10 / 5 = {calc.divide(10, 5)}")

    print("\n--- Testing Edge Case ---")
    try:
        calc.divide(10, 0)
    except ValueError as error:
        print(f"Error caught successfully: {error}")


if __name__ == "__main__":
    main()
