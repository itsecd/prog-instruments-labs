import math, sys, os


def Calculate(a, b, Operation):
    if Operation == '+':
        Result = a + b
    elif Operation == '-':
        Result = a - b
    elif Operation == '*':
        Result = a * b
    elif Operation == '/':
        if b != 0:
            Result = a / b
        else:
            return "Error: Division by zero"
    else:
        return "Error: Invalid operation"
    return Result


class Calculator:
    def __init__(self, Value=0):
        self.Value = Value
        self.History = []

    def Reset(self):
        self.Value = 0
        self.History.clear()

    def Add(self, x):
        self.Value = self.Value + x
        self.History.append(f"Added {x}")

    def Subtract(self, x):
        self.Value = self.Value - x
        self.History.append(f"Subtracted {x}")

    def Multiply(self, x):
        self.Value = self.Value * x
        self.History.append(f"Multiplied by {x}")

    def Divide(self, x):
        if x == 0:
            return "Cannot divide by zero"
        self.Value = self.Value / x
        self.History.append(f"Divided by {x}")

    def GetValue(self):
        return self.Value

    def GetHistory(self):
        return self.History

    def Power(self, exponent):
        self.Value = math.pow(self.Value, exponent)
        self.History.append(f"Powered to {exponent}")

    def SquareRoot(self):
        if self.Value < 0:
            return "Cannot calculate square root of negative number"
        self.Value = math.sqrt(self.Value)
        self.History.append("Square root calculated")


class AdvancedCalculator(Calculator):
    def __init__(self, Value=0):
        super().__init__(Value)
        self.Memory = 0
        self.Constants = {"pi": 3.14159, "e": 2.71828}

    def StoreToMemory(self):
        self.Memory = self.Value

    def RecallFromMemory(self):
        self.Value = self.Memory
        self.History.append("Recalled from memory")

    def ClearMemory(self):
        self.Memory = 0

    def AddConstant(self, name, value):
        self.Constants[name] = value

    def UseConstant(self, name):
        if name in self.Constants:
            self.Value = self.Constants[name]
            self.History.append(f"Used constant {name}")
        else:
            return f"Constant {name} not found"

    def Factorial(self):
        if self.Value < 0 or self.Value != int(self.Value):
            return "Cannot calculate factorial"
        result = 1
        for i in range(1, int(self.Value) + 1):
            result *= i
        self.Value = result
        self.History.append("Factorial calculated")


def ProcessUserInput():
    calc = Calculator()
    while True:
        print("Current value:", calc.GetValue())
        print("Available operations: +, -, *, /, p (power), s (sqrt), f (factorial), r (reset), h (history), q (quit)")
        operation = input("Enter operation: ")

        if operation == 'q':
            break
        elif operation == 'r':
            calc.Reset()
        elif operation == 'h':
            history = calc.GetHistory()
            for i, item in enumerate(history):
                print(f"{i + 1}. {item}")
        elif operation in ['+', '-', '*', '/']:
            try:
                num = float(input("Enter number: "))
                if operation == '+':
                    calc.Add(num)
                elif operation == '-':
                    calc.Subtract(num)
                elif operation == '*':
                    calc.Multiply(num)
                elif operation == '/':
                    result = calc.Divide(num)
                    if result:
                        print(result)
            except ValueError:
                print("Invalid number input")
        elif operation == 'p':
            try:
                exp = float(input("Enter exponent: "))
                calc.Power(exp)
            except ValueError:
                print("Invalid exponent")
        elif operation == 's':
            result = calc.SquareRoot()
            if result:
                print(result)
        elif operation == 'f':
            result = calc.Factorial()
            if result:
                print(result)
        else:
            print("Unknown operation")


def TestCalculator():
    # Test basic operations
    calc = Calculator(10)
    calc.Add(5)
    calc.Subtract(3)
    calc.Multiply(2)
    calc.Divide(4)
    assert calc.GetValue() == 6, f"Expected 6, got {calc.GetValue()}"

    # Test advanced calculator
    adv = AdvancedCalculator(5)
    adv.Factorial()
    assert adv.GetValue() == 120, f"Expected 120, got {adv.GetValue()}"

    adv.StoreToMemory()
    adv.Reset()
    adv.RecallFromMemory()
    assert adv.GetValue() == 120, f"Expected 120 after recall, got {adv.GetValue()}"

    print("All tests passed!")


class MathUtils:
    @staticmethod
    def IsPrime(n):
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def Fibonacci(n):
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

    @staticmethod
    def GCD(a, b):
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def LCM(a, b):
        return abs(a * b) // MathUtils.GCD(a, b)


def DemonstrateMathUtils():
    utils = MathUtils()
    print("Prime check for 17:", MathUtils.IsPrime(17))
    print("Prime check for 15:", MathUtils.IsPrime(15))
    print("10th Fibonacci number:", MathUtils.Fibonacci(10))
    print("GCD of 48 and 18:", MathUtils.GCD(48, 18))
    print("LCM of 12 and 18:", MathUtils.LCM(12, 18))


def Main():
    print("Welcome to the Calculator Demo!")
    print("Choose mode:")
    print("1. Interactive calculator")
    print("2. Run tests")
    print("3. Math utilities demo")

    choice = input("Enter choice (1-3): ")

    if choice == '1':
        ProcessUserInput()
    elif choice == '2':
        TestCalculator()
    elif choice == '3':
        DemonstrateMathUtils()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    Main()