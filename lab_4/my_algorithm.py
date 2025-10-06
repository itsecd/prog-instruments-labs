"""
This is my custom algorithm for finding a solution to a 24 Card (or to any card).
Hopefully it should be smarter than the brute-force algorithm.


Theory behind the algorithm:

When humans try to solve a 24 Card, they don't think in a brute-force way. Humans are smart.
When I try to solve a 24 Card, I think of FACTORS. Factors are very important. Normally, I'll try to find a factor
of 24 on my card and see if I can arrange the other 3 numbers to make the other factor (e.g. my card has a 4 in
it, can I use the other 3 numbers to make 4 so that I can multiply 4 * 6?).


Generalizing:

I figured that I could make this algorithm recursive. What I imagined was this: I'd pick one number from the
card, and then see if the other 3 numbers can make the other number that I need. What this inherently necessitates is
a general function which can answer this question:
        > Given n numbers, can you use arithmetic to arrive at an answer x?

Or more programmatically written as:
        > solution(array, x)    # array has size n


My algorithm in a nutshell:

Given n numbers (stored in array A), can you use arithmetic to arrive at x?

1.  If n = 1:
    a)  If A[0] = x, then yes
    b)  If A[0] != x, then no

2.  If n = 2:
    a)  Try multiplying the two numbers
    b)  Try adding the two numbers
    c)  If x >= 0
        i)  Try subtracting the smaller from the larger
        ii) Else try subtracting the larger from the smaller
    d)  Try dividing the larger by the smaller
    e)  If any of those work, then yes. If not, then no solution.

3.  Try adding all the numbers in A together.

4.  Try multiplying all the numbers in A together.

5.  If there are factors of x in A, pick one.
    * Prefer 1 as a factor of x, and then prefer smaller factors
    a)  See if the other n - 1 numbers can make the other factor of x
        i)  If so, then yes
        ii) If not, then pick another factor

6.  If there are no more factors of x in A, then for each number a in A:
    a)  SUBTRACT a from x
    * Prefer even numbers
    b)  See if the other n - 1 numbers can form the result
        i)  If so, then yes
        ii) If not, try the next number

7.  If that fails, then for each number a in A:
    a)  ADD a to x
    * Prefer Even Numbers
    b)  See if the other n - 1 numbers can form the result
        i)  If so, then yes
        ii) If not, try the next number

8.  If that fails, then for each number a in A:
    a)  MULTIPLY x by a
    * Prefer smaller numbers
    b)  See if the other n - 1 numbers can form the result
        i)  If so, then yes
        ii) If not, then no solution
"""

import sys
from math import sqrt
from typing import Optional

class OperationResult:
    value: float
    is_valid: bool = True
    error_message: str = ""

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
        self.op = op

    def _validate_operator(self):
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

# Constants for the different Operators
MUL = Operator('*')
ADD = Operator('+')
SUB = Operator('-')
DIV = Operator('/')
OPS = [MUL, ADD, SUB, DIV]

# The number of different combinations we've tried to find the answer
num_attempts = 0
current_attempt = Solution()
final_solution = None


def is_numeric(string: str) -> bool:
    """
    Checks if a string is numeric (as in alphanumeric without the alpha)
    :param string: The string we want to check
    :return: True if the string is numeric, False if not
    """
    return string.isalnum() and not string.isalpha()

def get_factors(n: int) -> list:
    """
    Returns all the factors of n.
    Code for this method can be attributed to user agf on StackExchange
    :param n: An integer
    :return: A list of whole-number factors of n
    """
    if n == 0:
        return []
    factors = set()
    for i in range(1, int(sqrt(abs(n))) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return list(factors)

def exclude(lst: list, value) -> list:
    """
    Returns lst excluding the first occurrence of value. Performs a deep copy in doing so.
    This is useful when we want to pass the n - 1 "other" numbers to another recursion of our solve() method.
    :param lst: The list we want to exclude value from
    :param value: The value we want to exclude
    :return: lst with the first occurrence of value excluded from it.
    """
    other_lst = lst.copy()  # Perform a deep copy so we can safely modify it
    if value in other_lst:
        other_lst.remove(value)
    return other_lst

def sort_evens_first(lst: list) -> list:
    """
    Sorts a list, putting the even numbers first (in ascending order)
    :param lst: The list we want to sort
    :return: The sorted list
    """
    evens = sorted([x for x in lst if x % 2 == 0])
    odds = sorted([x for x in lst if x % 2 == 1])
    return evens + odds


def is_correct(solution, value=24):
    """
    Checks if solution evaluates to value
    :param solution: The Solution instance that should be checked
    :param value: The number that we expect solution to evaluate to
    :return: True if solution evaluates to value, False if otherwise
    """
    return solution.evaluate() == value

def _solve_two_numbers(a: int, b: int, target: int, state: SolverState) -> Optional[Solution]:
    """
    Solves the case for exactly two numbers.
    """
    for left, right in [(a, b), (b, a)]:
        state.increment_attempts()
        if left * right == target:
            solution = Solution()
            solution.numbers = [left, right]
            solution.operations = [MUL]
            return solution
        
        state.increment_attempts()
        if left + right == target:
            solution = Solution()
            solution.numbers = [left, right]
            solution.operations = [ADD]
            return solution
        
        state.increment_attempts()
        if left - right == target:
            solution = Solution()
            solution.numbers = [left, right]
            solution.operations = [SUB]
            return solution
        
        state.increment_attempts()
        if right != 0 and abs(float(left) / right - target) < 1e-10:
            solution = Solution()
            solution.numbers = [left, right]
            solution.operations = [DIV]
            return solution
    
    return

def _check_total_sum(numbers: list, target: int, state: SolverState) -> Optional[Solution]:
    """
    Checks if sum of all numbers equals target.
    """
    state.increment_attempts()
    if sum(numbers) == target:
        solution = Solution()
        solution.numbers = numbers
        solution.operations = [ADD] * (len(numbers) - 1)
        return solution
    return

def _check_total_product(numbers: list, target: int, state: SolverState) -> Optional[Solution]:
    """
    Checks if product of all numbers equals target.
    """
    state.increment_attempts()
    product = 1
    for num in numbers:
        product *= num
    if product == target:
        solution = Solution()
        solution.numbers = numbers
        solution.operations = [MUL] * (len(numbers) - 1)
        return solution
    return

def _try_factoring_approach(numbers: list, target: int, state: SolverState) -> Optional[Solution]:
    """
    Tries to solve using factoring approach.
    """
    if target <= 0:
        return None

    factors = get_factors(target)
    factors_in_list = [num for num in numbers if num in factors]
    
    for factor in sorted(factors_in_list):
        other_factor = target // factor
        other_numbers = exclude(numbers, factor)
        solution = solve(other_numbers, other_factor, state)
        if solution:
            solution.numbers.append(factor)
            solution.operations.append(MUL)
            return solution
    
    return

def _try_arithmetic_approaches(numbers: list, target: int, state: SolverState) -> Optional[Solution]:
    """
    Tries addition, subtraction, and division approaches.
    """
    addition_solution = _try_addition_approach(numbers, target, state)
    if addition_solution:
        return addition_solution
        
    subtraction_solution = _try_subtraction_approach(numbers, target, state)
    if subtraction_solution:
        return subtraction_solution
        
    division_solution = _try_division_approach(numbers, target, state)
    return division_solution

def _try_addition_approach(numbers: list, target: int, state: SolverState) -> Optional[Solution]:
    """Tries solving using addition strategy."""
    numbers_sorted = sort_evens_first(numbers)
    for num in numbers_sorted:
        other_numbers = exclude(numbers, num)
        solution = solve(other_numbers, target - num, state)
        if solution:
            solution.numbers.append(num)
            solution.operations.append(ADD)
            return solution
    return

def _try_subtraction_approach(numbers: list, target: int, state: SolverState) -> Optional[Solution]:
    """Tries solving using subtraction strategy."""
    numbers_sorted = sort_evens_first(numbers)
    for num in numbers_sorted:
        other_numbers = exclude(numbers, num)
        solution = solve(other_numbers, target + num, state)
        if solution:
            solution.numbers.append(num)
            solution.operations.append(SUB)
            return solution
    return

def _try_division_approach(numbers: list, target: int, state: SolverState) -> Optional[Solution]:
    """Tries solving using division strategy."""
    for num in sorted(numbers):
        if num == 0:
            continue
        other_numbers = exclude(numbers, num)
        solution = solve(other_numbers, target * num, state)
        if solution:
            solution.numbers.append(num)
            solution.operations.append(DIV)
            return solution
    return

def solve(numbers: list, target: int, state: SolverState) -> Optional[Solution]:
    """
    Uses arithmetic (*, +, -, /) to arrive at value, using my custom recursive algorithm
    :param numbers: The list of numbers we're going to use to arrive at value
    :param value: The value that we want to arrive at using all of the
    :return: If solvable, returns a Solution instance. If not, returns False.
    """
    state.increment_attempts()
    
    if not numbers:
        return
        
    n = len(numbers)
    
    if n == 1:
        if numbers[0] == target:
            solution = Solution()
            solution.numbers = [target]
            solution.operations = []
            return solution
        else:
            return

    if n == 2:
        return _solve_two_numbers(numbers[0], numbers[1], target, state)

    total_sum_solution = _check_total_sum(numbers, target, state)
    if total_sum_solution:
        return total_sum_solution

    total_product_solution = _check_total_product(numbers, target, state)
    if total_product_solution:
        return total_product_solution

    factoring_solution = _try_factoring_approach(numbers, target, state)
    if factoring_solution:
        return factoring_solution

    return _try_arithmetic_approaches(numbers, target, state)


def solve_card(card: list, target: int = 24) -> tuple[Optional[Solution], int]:
    """
    This method solves the 24 Card using my custom algorithm
    :param card: A list representing the 24 Card
    : param target: Target value (default 24)
    :return: Tuple of (solution, attempts_count)
    """
    state = SolverState()
    solution = solve(card, target, state)
    return solution, state.attempts_count


def get_card_from_user() -> list:
    """Gets card numbers from command line or user input."""
    if len(sys.argv) > 1:
        return [int(arg) for arg in sys.argv[1:] if is_numeric(arg)]
    else:
        user_input = input("Please enter 4 numbers separated by a space: ")
        return [int(num_str) for num_str in user_input.split() if is_numeric(num_str)]


if __name__ == "__main__":
    try:
        card = get_card_from_user()
        
        if len(card) != 4:
            raise ValueError("Length not equal 4!")

        print(f"Solving for card: {card}")
        
        solution, attempts = solve_card(card)
        
        if solution:
            try:
                if solution.is_correct():
                    print(f"Solution: {solution}")
                    print(f"Verification: {solution.evaluate()}")
            except Exception as e:
                print("Error in solution evaluation: {e}")
            
        else:
            print("No solution found!")
        
        print(f"Number of attempts: {attempts}")

    except ValueError as e:
        print(f"Input error: Please enter valid numbers - {e}")
    except Exception as e:
        print(f"Something went wrong: {e}")