import math
from typing import List, Union, Optional


def calculate(a: float, b: float, operation: str) -> Union[float, str]:
    if operation == '+':
        result: float = a + b
    elif operation == '-':
        result: float = a - b
    elif operation == '*':
        result: float = a * b
    elif operation == '/':
        if b != 0:
            result: float = a / b
        else:
            return "Error: Division by zero"
    else:
        return "Error: Invalid operation"
    return result


class Calculator:
    def __init__(self, value: float = 0) -> None:
        self.value: float = value
        self.history: List[str] = []

    def reset(self) -> None:
        self.value = 0
        self.history.clear()

    def add(self, x: float) -> None:
        self.value = self.value + x
        self.history.append(f"added {x}")

    def subtract(self, x: float) -> None:
        self.value = self.value - x
        self.history.append(f"subtracted {x}")

    def multiply(self, x: float) -> None:
        self.value = self.value * x
        self.history.append(f"Multiplied by {x}")

    def divide(self, x: float) -> Optional[str]:
        if x == 0:
            return "Cannot divide by zero"
        self.value = self.value / x
        self.history.append(f"divided by {x}")
        return None

    def get_value(self) -> float:
        return self.value

    def get_history(self) -> List[str]:
        return self.history

    def power(self, exponent: float) -> None:
        self.value = math.pow(self.value, exponent)
        self.history.append(f"Powered to {exponent}")

    def square_root(self) -> Optional[str]:
        if self.value < 0:
            return "Cannot calculate square root of negative number"
        self.value = math.sqrt(self.value)
        self.history.append("Square root calculated")
        return None


class AdvancedCalculator(Calculator):
    def __init__(self, value: float = 0) -> None:
        super().__init__(value)
        self.memory: float = 0
        self.constants: dict[str, float] = {"pi": 3.14159, "e": 2.71828}

    def store_to_memory(self) -> None:
        self.memory = self.value

    def recall_from_memory(self) -> None:
        self.value = self.memory
        self.history.append("Recalled from memory")

    def clear_memory(self) -> None:
        self.memory = 0

    def add_constant(self, name: str, value: float) -> None:
        self.constants[name] = value

    def use_constant(self, name: str) -> Optional[str]:
        if name in self.constants:
            self.value = self.constants[name]
            self.history.append(f"Used constant {name}")
            return None
        else:
            return f"Constant {name} not found"

    def factorial(self) -> Optional[str]:
        if self.value < 0 or self.value != int(self.value):
            return "Cannot calculate factorial"
        result: int = 1
        for i in range(1, int(self.value) + 1):
            result *= i
        self.value = result
        self.history.append("Factorial calculated")
        return None


def process_user_input() -> None:
    calc: Calculator = Calculator()
    while True:
        print("Current value:", calc.get_value())
        print("Available operations: +, -, *, /, p (power), s (sqrt), f (factorial), r (reset), h (history), q (quit)")
        operation: str = input("Enter operation: ")

        if operation == 'q':
            break
        elif operation == 'r':
            calc.reset()
        elif operation == 'h':
            history: List[str] = calc.get_history()
            for i, item in enumerate(history):
                print(f"{i + 1}. {item}")
        elif operation in ['+', '-', '*', '/']:
            try:
                num: float = float(input("Enter number: "))
                if operation == '+':
                    calc.add(num)
                elif operation == '-':
                    calc.subtract(num)
                elif operation == '*':
                    calc.multiply(num)
                elif operation == '/':
                    result: Optional[str] = calc.divide(num)
                    if result:
                        print(result)
            except ValueError:
                print("Invalid number input")
        elif operation == 'p':
            try:
                exp: float = float(input("Enter exponent: "))
                calc.power(exp)
            except ValueError:
                print("Invalid exponent")
        elif operation == 's':
            result: Optional[str] = calc.square_root()
            if result:
                print(result)
        elif operation == 'f':
            result: Optional[str] = calc.factorial()
            if result:
                print(result)
        else:
            print("Unknown operation")


def test_calculator() -> None:
    # Test basic operations
    calc: Calculator = Calculator(10)
    calc.add(5)
    calc.subtract(3)
    calc.multiply(2)
    calc.divide(4)
    assert calc.get_value() == 6, f"Expected 6, got {calc.get_value()}"

    # Test advanced calculator
    adv: AdvancedCalculator = AdvancedCalculator(5)
    adv.factorial()
    assert adv.get_value() == 120, f"Expected 120, got {adv.get_value()}"

    adv.store_to_memory()
    adv.reset()
    adv.recall_from_memory()
    assert adv.get_value() == 120, f"Expected 120 after recall, got {adv.get_value()}"

    print("All tests passed!")


class MathUtils:
    @staticmethod
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def fibonacci(n: int) -> int:
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a: int = 0
            b: int = 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

    @staticmethod
    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def lcm(a: int, b: int) -> int:
        return abs(a * b) // MathUtils.gcd(a, b)


def demonstrate_math_utils() -> None:
    print("Prime check for 17:", MathUtils.is_prime(17))
    print("Prime check for 15:", MathUtils.is_prime(15))
    print("10th Fibonacci number:", MathUtils.fibonacci(10))
    print("GCD of 48 and 18:", MathUtils.gcd(48, 18))
    print("LCM of 12 and 18:", MathUtils.lcm(12, 18))


def main() -> None:
    print("Welcome to the Calculator Demo!")
    print("Choose mode:")
    print("1. Interactive calculator")
    print("2. Run tests")
    print("3. Math utilities demo")

    choice: str = input("Enter choice (1-3): ")

    if choice == '1':
        process_user_input()
    elif choice == '2':
        test_calculator()
    elif choice == '3':
        demonstrate_math_utils()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()