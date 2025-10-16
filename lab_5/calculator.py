"""imports the os and time modules from the Python Standard Library.
The os module provides a way of using operating system dependent functionality,
like reading or writing to the file system.
The time module provides various time-related functions, like getting the current
time or pausing the execution of the script."""

import os
import sys
import time
import logging
from datetime import datetime

LOG_FILENAME = 'calculator.log'

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d %(funcName)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[logging.FileHandler(LOG_FILENAME,encoding="utf-8"),
                    logging.StreamHandler()])

logger = logging.getLogger(__name__)


def addition():
    """This function asks the user to enter a series of numbers separated by spaces.
    It then adds all the numbers together and returns the result.
    """
    try:
        nums = list(map(float, input("Enter all numbers separated by space: ").split()))
        logger.info(f"Addiction called with{nums}")
        result = sum(nums)
        logger.debug(f"Result: {result}")
        return result
    except ValueError as e:
        logger.error(f"Invalid input for addition{e}")
        return "Invalid input"

def subtraction():
    """This function asks the user to enter two numbers.
    It then subtracts the second number from the first and returns the result."""
    try:
        n1 = float(input("Enter first number: "))
        n2 = float(input("Enter second number: "))
        logger.info(f"Subtraction called with{n1}, {n2}")
        result = n1 - n2
        logger.debug(f"Result: {result}")
        return result
    except ValueError as e:
        logger.error(f"Invalid input for subtraction{e}")
        return "Invalid input"

def multiplication():
    """Function asks user to enter a series of numbers separated by spaces.
    Then multiply all the numbers together and returns the result."""
    try:
        nums = list(map(float, input("Enter all numbers separated by space: ").split()))
        logger.info(f"Multiplication called with{nums}")
        result = 1
        for num in nums:
            result *= num
        logger.debug(f"Result: {result}")
        return result
    except ValueError as e:
        logger.error(f"Invalid input for multiplication{e}")
        return "invalid input"

def division():
    """Function divide two numbers"""
    try:
        n1 = float(input("Enter first number: "))
        n2 = float(input("Enter second number: "))
        logger.info(f"Division called with{n1}, {n2}")
        if n2 == 0:
            logger.warning(f"Division by zero attempted")
            return "Invalid entry"
        result = n1 / n2
        logger.debug(f"Result: {result}")
        return result
    except ValueError as e:
       logger.error(f"Invalid input for division{e}")
       return "invalid input"

def average():
    """This function takes space separated number series and then convert it to a list.
    Then calculates the average of that list of numbers."""
    try:
        nums = list(map(float, input("Enter all numbers separated by space: ").split()))
        logger.info(f"Average called with{nums}")
        result = sum(nums) / len(nums)
        logger.debug(f"Result: {result}")
        return result
    except ValueError as e:
        logger.error(f"Error with average{e}")
        return "Invalid input"

def factorial(num):
    """
    Function to calculate the factorial of a number.

    Takes a number as an argument, calculates the factorial of the number,
    and returns the result.
    """
    try:
        if num < 0:
            logger.warning(f"Factorial called with negative number: {num}")
            return "Invalid entry"
        result = 1
        for i in range(1, num+1):
            result *= i
        logger.debug(f"Result: {result}")
        return result
    except ValueError as e:
        logger.error(f"Error with factorial{e}")
        return "Invalid input"

def complex_arithmetic():
    """
    Function to execute complex arithmetic operations such as addition, subtraction, multiplication, and division.

    Asks the user to choose the operation and input the complex numbers as real and imaginary parts,
    performs the operation, and returns the result.
    """


    print("Enter '1' for addition, '2' for subtraction, '3' for multiplication, '4' for division")
    choice = input("Enter your choice: ")
    logger.info(f"Complex arithmetic operation selected: {choice}")
    try:
        nums = list(map(float, input("Enter real and imaginary parts separated by space: ").split()))
        if len(nums) not in [4, 6]:
            logger.warning("Unexpected number of inputs for complex operation")
        a, b, c, d = nums[:4]
        if choice == "1":
            result = f"{a + c} + i{b + d}"
        elif choice == "2":
            result = f"{a - c} + i{b - d}"
        elif choice == "3":
            real = a * c - b * d
            imag = a * d + b * c
            result = f"{real} + i{imag}"
        elif choice == "4":
            denom = c ** 2 + d ** 2
            if denom == 0:
                logger.warning("Division by zero in complex arithmetic")
                return "Invalid (division by zero)"
            real = (a * c + b * d) / denom
            imag = (b * c - a * d) / denom
            result = f"{real} + i{imag}"
        else:
            logger.warning(f"Invalid complex operation choice: {choice}")
            return "Invalid choice"
        logger.debug(f"Complex operation result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in complex arithmetic: {e}")
        return "Invalid input"


def binomial(num):
    """
    Function to calculate the binomial coefficient.

    Takes two numbers as arguments, calculates the binomial coefficient using the formula n!/(k!(n-k)!),
    and returns the result.
    """
    try:
        if num[0] < num[1]:
            logger.warning(f"Invalid binomial input n={num[0]}, k={num[1]}")
            return "Invalid entry"
        result = factorial(int(num[0])) / (factorial(int(num[1])) * factorial(int(num[0] - num[1])))
        logger.debug(f"Binomial({num[0]}, {num[1]}) = {result}")
        return result
    except Exception as e:
        logger.error(f"Error in binomial({num}): {e}")
        return "Invalid input"

logger.info("Program started")
c = 0

while c !="-1":
    print("=== Menu ===")
    print("1: Addition | 2: Subtraction | 3: Multiplication | 4: Division")
    print("5: Average | 6: Factorial | 7: Complex arithmetic | 8: Binomial")
    print("-1: Exit\n")

    c = input("Enter your choice: ")
    logger.info(f"Choice: {c}")

    if c == "1":
        result = addition()
    elif c == "2":
        result = subtraction()
    elif c == "3":
        result = multiplication()
    elif c == "4":
        result = division()
    elif c == "5":
        result = average()
    elif c == "6":
        num = int(input("Enter a number: "))
        result = factorial(num)
    elif c == "7":
        result = complex_arithmetic()
    elif c == "8":
        num = list(map(int, input("Enter n and k separated by space: ").split()))
        result = binomial(num)
    elif c == "-1":
        logger.info("User exited the calculator.")
        print("Thank you for using the calculator!")
        break
    else:
        logger.warning(f"Invalid menu choice: {c}")
        print("Invalid option!")
        continue

    print(f"The result is: {result}")
    logger.info(f"Operation result: {result}")
    time.sleep(1)
    os.system("cls||clear")

logger.info("Calculator session ended.")