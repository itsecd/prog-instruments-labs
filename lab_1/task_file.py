import math
from typing import List, Union, Optional


def calculate(a: float, b: float, operation: str) -> Union[float, str]:
    """
    Perform basic arithmetic operations on two numbers.

    Args:
        a: First operand
        b: Second operand
        operation: Arithmetic operation (+, -, *, /)

    Returns:
        Result of the operation or error message

    Raises:
        ValueError: If operation is invalid or division by zero
    """
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
    """A simple calculator with basic operations and history tracking."""

    def __init__(self, value: float = 0) -> None:
        """
        Initialize the calculator with an initial value.

        Args:
            value: Starting value for the calculator
        """
        self.value: float = value
        self.history: List[str] = []

    def reset(self) -> None:
        """Reset the calculator value to zero and clear history."""
        self.value = 0
        self.history.clear()

    def add(self, x: float) -> None:
        """
        Add a number to the current value.

        Args:
            x: Number to add
        """
        self.value = self.value + x
        self.history.append(f"added {x}")

    def subtract(self, x: float) -> None:
        """
        Subtract a number from the current value.

        Args:
            x: Number to subtract
        """
        self.value = self.value - x
        self.history.append(f"subtracted {x}")

    def multiply(self, x: float) -> None:
        """
        Multiply the current value by a number.

        Args:
            x: Number to multiply by
        """
        self.value = self.value * x
        self.history.append(f"Multiplied by {x}")

    def divide(self, x: float) -> Optional[str]:
        """
        Divide the current value by a number.

        Args:
            x: Number to divide by

        Returns:
            Error message if division by zero, None otherwise
        """
        if x == 0:
            return "Cannot divide by zero"
        self.value = self.value / x
        self.history.append(f"divided by {x}")
        return None

    def get_value(self) -> float:
        """
        Get the current value of the calculator.

        Returns:
            Current calculator value
        """
        return self.value

    def get_history(self) -> List[str]:
        """
        Get the operation history.

        Returns:
            List of operation descriptions
        """
        return self.history

    def power(self, exponent: float) -> None:
        """
        Raise the current value to a power.

        Args:
            exponent: The exponent to raise to
        """
        self.value = math.pow(self.value, exponent)
        self.history.append(f"Powered to {exponent}")

    def square_root(self) -> Optional[str]:
        """
        Calculate the square root of the current value.

        Returns:
            Error message for negative numbers, None otherwise
        """
        if self.value < 0:
            return "Cannot calculate square root of negative number"
        self.value = math.sqrt(self.value)
        self.history.append("Square root calculated")
        return None


class AdvancedCalculator(Calculator):
    """An advanced calculator with memory and constants functionality."""

    def __init__(self, value: float = 0) -> None:
        """
        Initialize advanced calculator with memory and constants.

        Args:
            value: Starting value for the calculator
        """
        super().__init__(value)
        self.memory: float = 0
        self.constants: dict[str, float] = {"pi": 3.14159, "e": 2.71828}

    def store_to_memory(self) -> None:
        """Store the current value to memory."""
        self.memory = self.value

    def recall_from_memory(self) -> None:
        """Recall the value from memory to the current value."""
        self.value = self.memory
        self.history.append("Recalled from memory")

    def clear_memory(self) -> None:
        """Clear the memory by setting it to zero."""
        self.memory = 0

    def add_constant(self, name: str, value: float) -> None:
        """
        Add a custom constant to the calculator.

        Args:
            name: Name of the constant
            value: Value of the constant
        """
        self.constants[name] = value

    def use_constant(self, name: str) -> Optional[str]:
        """
        Use a predefined constant as the current value.

        Args:
            name: Name of the constant to use

        Returns:
            Error message if constant not found, None otherwise
        """
        if name in self.constants:
            self.value = self.constants[name]
            self.history.append(f"Used constant {name}")
            return None
        else:
            return f"Constant {name} not found"

    def factorial(self) -> Optional[str]:
        """
        Calculate factorial of the current integer value.

        Returns:
            Error message for invalid input, None otherwise
        """
        if self.value < 0 or self.value != int(self.value):
            return "Cannot calculate factorial"
        result: int = 1
        for i in range(1, int(self.value) + 1):
            result *= i
        self.value = result
        self.history.append("Factorial calculated")
        return None


def process_user_input() -> None:
    """Process user input for interactive calculator mode."""
    calc: Calculator = Calculator()
    while True:
        print("Current value:", calc.get_value())
        print("Available operations: +, -, *, /,"
              " p (power), s (sqrt), f (factorial),"
              " r (reset), h (history), q (quit)")
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
    """Run tests to verify calculator functionality."""
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
    """A collection of mathematical utility functions."""

    @staticmethod
    def is_prime(n: int) -> bool:
        """
        Check if a number is prime.

        Args:
            n: Number to check

        Returns:
            True if prime, False otherwise
        """
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def fibonacci(n: int) -> int:
        """
        Calculate the nth Fibonacci number.

        Args:
            n: Position in Fibonacci sequence

        Returns:
            The nth Fibonacci number
        """
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
        """
        Calculate The Greatest Common Divisor of two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            GCD of a and b
        """
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def lcm(a: int, b: int) -> int:
        """
        Calculate The Least Common Multiple of two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            LCM of a and b
        """
        return abs(a * b) // MathUtils.gcd(a, b)


def demonstrate_math_utils() -> None:
    """Demonstrate the mathematical utility functions."""
    print("Prime check for 17:", MathUtils.is_prime(17))
    print("Prime check for 15:", MathUtils.is_prime(15))
    print("10th Fibonacci number:", MathUtils.fibonacci(10))
    print("GCD of 48 and 18:", MathUtils.gcd(48, 18))
    print("LCM of 12 and 18:", MathUtils.lcm(12, 18))


def main() -> None:
    """Main function to run the calculator demo application."""
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