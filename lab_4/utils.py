from typing import Optional


class OperationResult:
    """Represents the result of an arithmetic operation with validation."""
    def __init__(self, value: float = 0.0, 
                 is_valid: bool = True, error_message: str = ""):
        """
        Initialize OperationResult.
        :param value: The numerical result of the operation
        :param is_valid: Whether the operation was successful
        :param error_message: Description of any error that occurred
        """
        self.value = value
        self.is_valid = is_valid
        self.error_message = error_message

class Operator(object):
    """
    Represents an operator ('*', '+', '-', '/') used in solving a 24 Card.
    """
    _operations = {
        '*': lambda left, right: OperationResult(left * right),
        '+': lambda left, right: OperationResult(left + right),
        '-': lambda left, right: OperationResult(left - right),
        '/': lambda left, right: (
            OperationResult(float(left) / right) if right != 0 
            else OperationResult(0, False, "Division by zero")
        )
    }

    def __init__(self, op):
        """
        Initialize operator with given symbol.
        :param op: Operator symbol ('*', '+', '-', '/')
        """
        self.op = op

    def _validate_operator(self):
        """Validate that operator symbol is supported."""
        if self.op not in self._operations:
            raise ValueError(f"Invalid operator: {self.op}")

    def evaluate(self, left, right):
        """
        Evaluates the result of multiplying/adding/subtracting/dividing left and right
        :param left: The left operand
        :param right: The right operand
        :return: The result of executing the operator on the two operands
        """
        return self._operations[self.op](left, right)

    def __repr__(self):
        return str(self.op)

class Solution(object):
    """
    Represents a potential solution to a 24 Card.
    Has an array of 4 numbers and an array of 3 operations.
    A Solution does not necessarily have to be correct.
    """

    def __init__(self, numbers: Optional[list[float]] = None, 
                 operations: Optional[list[Operator]] = None):
        """
        Initialize Solution with numbers and operations.
        :param numbers: List of numbers in the solution
        :param operations: List of operations between numbers
        """
        self.numbers = numbers or []
        self.operations = operations or []

    def evaluate(self):
        """
        Evaluates the result of this Solution.
        Executes the 3 operations (in order) on the 4 numbers (in order).
        num1 <op1> num2 <op2> num3 <op3> num4
        :return: The result of evaluating the Solution.
        """
        result = self.numbers[0]
        for i in range(1, len(self.numbers)):
            left = result
            right = self.numbers[i]
            operator = self.operations[i - 1]
            result = operator.evaluate(left, right)
        return result
    
    def is_correct(self, target: float = 24) -> bool:
        """Checks if solution evaluates to target value."""
        return abs(self.evaluate() - target) < 1e-10


    def __repr__(self):
        """
        Makes a human-readable string to represent this Solution
        :return The string representation of this Solution
        """
        result = str(self.numbers[0])
        for i in range(1, len(self.numbers)):
            result += (f" {self.operations[i - 1]} {self.numbers[i]}")
        return result


class SolverState:
    """Tracks the state of the solver during computation."""
        
    def __init__(self) -> None:
        """Initialize solver state."""
        self.attempts_count: int = 0
        self.current_solution: Optional[Solution] = None

    def increment_attempts(self) -> None:
        """Increment the attempts counter."""
        self.attempts_count += 1

    def set_current_solution(self, solution: Solution) -> None:
        """Set the current solution being evaluated."""
        self.current_solution = solution