import sys

from math import sqrt
from typing import Optional

from utils import Operator, SolverState, Solution

class TwentyFourSolver:
    """Main solver class for 24 Card game that encapsulates all solving logic"""
    # Constants for the different Operators
    MUL = Operator('*')
    ADD = Operator('+')
    SUB = Operator('-')
    DIV = Operator('/')
    OPS = [MUL, ADD, SUB, DIV]

    def __init__(self):
        """Initialize the TwentyFourSolver."""
        pass

    def is_numeric(self, string: str) -> bool:
        """
        Checks if a string is numeric (as in alphanumeric without the alpha)
        :param string: The string we want to check
        :return: True if the string is numeric, False if not
        """
        return string.isalnum() and not string.isalpha()

    def get_factors(self, n: int) -> list:
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

    def exclude(self, lst: list, value) -> list:
        """
        Returns lst excluding the first occurrence of value. Performs a deep copy in doing so.
        This is useful when we want to pass the n - 1 "other" numbers to another recursion of our solve() method.
        :param lst: The list we want to exclude value from
        :param value: The value we want to exclude
        :return: lst with the first occurrence of value excluded from it.
        """
        other_lst = lst.copy()
        if value in other_lst:
            other_lst.remove(value)
        return other_lst

    def sort_evens_first(self, lst: list) -> list:
        """
        Sorts a list, putting the even numbers first (in ascending order)
        :param lst: The list we want to sort
        :return: The sorted list
        """
        evens = sorted([x for x in lst if x % 2 == 0])
        odds = sorted([x for x in lst if x % 2 == 1])
        return evens + odds


    def is_correct(self, solution, value: float = 24):
        """
        Checks if solution evaluates to value
        :param solution: The Solution instance that should be checked
        :param value: The number that we expect solution to evaluate to
        :return: True if solution evaluates to value, False if otherwise
        """
        return solution.evaluate() == value

    
    def _solve_two_numbers(self, a: int, 
                           b: int, target: int, state: SolverState) -> Optional[Solution]:
        """
        Solves the case for exactly two numbers.
        :param a: First number
        :param b: Second number
        :param target: Target value
        :param state: Solver state for tracking attempts
        :return: Solution if found, None otherwise
        """
        for left, right in [(a, b), (b, a)]:
            state.increment_attempts()
            if left * right == target:
                solution = Solution()
                solution.numbers = [left, right]
                solution.operations = [self.MUL]
                return solution
            
            state.increment_attempts()
            if left + right == target:
                solution = Solution()
                solution.numbers = [left, right]
                solution.operations = [self.ADD]
                return solution
            
            state.increment_attempts()
            if left - right == target:
                solution = Solution()
                solution.numbers = [left, right]
                solution.operations = [self.SUB]
                return solution
            
            state.increment_attempts()
            if right != 0 and abs(float(left) / right - target) < 1e-10:
                solution = Solution()
                solution.numbers = [left, right]
                solution.operations = [self.DIV]
                return solution
        
        return

    
    def _check_total_sum(self, numbers: list, 
                         target: int, state: SolverState) -> Optional[Solution]:
        """
        Checks if sum of all numbers equals target.
        :param numbers: List of numbers
        :param target: Target value
        :param state: Solver state for tracking attempts
        :return: Solution if sum equals target, None otherwise
        """
        state.increment_attempts()
        if sum(numbers) == target:
            solution = Solution()
            solution.numbers = numbers
            solution.operations = [self.ADD] * (len(numbers) - 1)
            return solution
        return

    
    def _check_total_product(self, numbers: list, 
                             target: int, state: SolverState) -> Optional[Solution]:
        """
        Checks if product of all numbers equals target.
        :param numbers: List of numbers
        :param target: Target value
        :param state: Solver state for tracking attempts
        :return: Solution if product equals target, None otherwise
        """
        state.increment_attempts()
        product = 1
        for num in numbers:
            product *= num
        if product == target:
            solution = Solution()
            solution.numbers = numbers
            solution.operations = [self.MUL] * (len(numbers) - 1)
            return solution
        return

    
    def _try_factoring_approach(self, numbers: list, 
                                target: int, state: SolverState) -> Optional[Solution]:
        """
        Tries to solve using factoring approach.
        :param numbers: List of numbers
        :param target: Target value
        :param state: Solver state for tracking attempts
        :return: Solution if found, None otherwise
        """
        if target <= 0:
            return None

        factors = self.get_factors(target)
        factors_in_list = [num for num in numbers if num in factors]
        
        for factor in sorted(factors_in_list):
            other_factor = target // factor
            other_numbers = self.exclude(numbers, factor)
            solution = self.solve(other_numbers, other_factor, state)
            if solution:
                solution.numbers.append(factor)
                solution.operations.append(self.MUL)
                return solution
        
        return

    

    def _try_addition_approach(self, numbers: list, 
                               target: int, state: SolverState) -> Optional[Solution]:
        """
        Tries solving using addition strategy.
        :param numbers: List of numbers
        :param target: Target value
        :param state: Solver state for tracking attempts
        :return: Solution if found, None otherwise
        """
        numbers_sorted = self.sort_evens_first(numbers)
        for num in numbers_sorted:
            other_numbers = self.exclude(numbers, num)
            solution = self.solve(other_numbers, target - num, state)
            if solution:
                solution.numbers.append(num)
                solution.operations.append(self.ADD)
                return solution
        return

    
    def _try_subtraction_approach(self, numbers: list, 
                                  target: int, state: SolverState) -> Optional[Solution]:
        """
        Tries solving using subtraction strategy.
        :param numbers: List of numbers
        :param target: Target value
        :param state: Solver state for tracking attempts
        :return: Solution if found, None otherwise
        """
        numbers_sorted = self.sort_evens_first(numbers)
        for num in numbers_sorted:
            other_numbers = self.exclude(numbers, num)
            solution = self.solve(other_numbers, target + num, state)
            if solution:
                solution.numbers.append(num)
                solution.operations.append(self.SUB)
                return solution
        return

    
    def _try_division_approach(self, numbers: list, 
                               target: int, state: SolverState) -> Optional[Solution]:
        """
        Tries solving using division strategy.
        :param numbers: List of numbers
        :param target: Target value
        :param state: Solver state for tracking attempts
        :return: Solution if found, None otherwise
        """
        for num in sorted(numbers):
            if num == 0:
                continue
            other_numbers = self.exclude(numbers, num)
            solution = self.solve(other_numbers, target * num, state)
            if solution:
                solution.numbers.append(num)
                solution.operations.append(self.DIV)
                return solution
        return
    

    def _try_arithmetic_approaches(self, numbers: list, 
                                   target: int, state: SolverState) -> Optional[Solution]:
        """
        Tries addition, subtraction, and division approaches.
        :param numbers: List of numbers
        :param target: Target value
        :param state: Solver state for tracking attempts
        :return: Solution if found, None otherwise
        """
        addition_solution = self._try_addition_approach(numbers, target, state)
        if addition_solution:
            return addition_solution
            
        subtraction_solution = self._try_subtraction_approach(numbers, target, state)
        if subtraction_solution:
            return subtraction_solution
            
        division_solution = self._try_division_approach(numbers, target, state)
        return division_solution
    

    def solve(self, numbers: list, target: int, state: SolverState) -> Optional[Solution]:
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
            return self._solve_two_numbers(numbers[0], numbers[1], target, state)

        total_sum_solution = self._check_total_sum(numbers, target, state)
        if total_sum_solution:
            return total_sum_solution

        total_product_solution = self._check_total_product(numbers, target, state)
        if total_product_solution:
            return total_product_solution

        factoring_solution = self._try_factoring_approach(numbers, target, state)
        if factoring_solution:
            return factoring_solution

        return self._try_arithmetic_approaches(numbers, target, state)


    def solve_card(self, card: list, target: int = 24) -> tuple[Optional[Solution], int]:
        """
        This method solves the 24 Card using my custom algorithm
        :param card: A list representing the 24 Card
        : param target: Target value (default 24)
        :return: Tuple of (solution, attempts_count)
        """
        state = SolverState()
        solution = self.solve(card, target, state)
        return solution, state.attempts_count


    def get_card_from_user(self) -> list:
        """Gets card numbers from command line or user input."""
        if len(sys.argv) > 1:
            return [int(arg) for arg in sys.argv[1:] if self.is_numeric(arg)]
        else:
            user_input = input("Please enter 4 numbers separated by a space: ")
            return [int(num_str) for num_str in user_input.split() if self.is_numeric(num_str)]
        
    
    def run(self) -> None:
        """
        Runs the main game loop with exception handling.
        """
        try:
            card = self.get_card_from_user()
            
            if len(card) != 4:
                raise ValueError("Length not equal 4!")

            print(f"Solving for card: {card}")
            
            solution, attempts = self.solve_card(card)
            
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


if __name__ == "__main__":
    solver = TwentyFourSolver()
    solver.run()