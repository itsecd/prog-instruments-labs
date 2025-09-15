# This is like making a package.lock file for npm package.
# Yes, I should be making it.
__author__ = "Nitkarsh Chourasia"
__version__ = "0.0.0"  # SemVer # Understand more about it
__license__ = "MIT"  # Understand more about it
# Want to make it open source but how to do it?
# Program to make a simple calculator
# Will have to extensively work on Jarvis and local_document
# and MongoDb and Redis and JavaScript and CSS and DOM manipulation
# to understand it.
# Will have to study maths to understand it more better.
# How can I market gtts? Like showing used google's api?
# This is how can I market it?
# Project description? What will be the project description?

from gtts import gTTS
from pygame import mixer, time
from io import BytesIO
from pprint import pprint


# Find the best of best extensions for the auto generation of
# the documentation parts.
# For your favourite languages like JavaScript, Python ,etc,...
# Should be able to print date and time too.
# Should use voice assistant for specially abled people.
# A fully personalised calculator.
# voice_assistant on/off , setting bool value to true or false

# Is the operations valid?


# Validation checker
class Calculator:
    def __init__(self):
        self.take_inputs()

    def add(self):
        """Get the sum of numbers.

        Returns:
            float: The sum of num1 and num2.
        """
        return self.num1 + self.num2

    def sub(self):
        """Get the difference of numbers.

        Returns:
            float: The difference of num1 and num2.
        """
        return self.num1 - self.num2

    def multi(self):
        """Get the product of numbers.

        Returns:
            float: The product of num1 and num2.
        """
        return self.num1 * self.num2

    def div(self):
        """Get the quotient of numbers.

        Returns:
            float: The quotient of num1 divided by num2.
        """
        # What do we mean by quotient?
        return self.num1 / self.num2

    def power(self):
        """Get the power of numbers.

        Returns:
            float: num1 raised to the power of num2.
        """
        return self.num1 ** self.num2  # ИСПРАВЛЕНО: добавлены пробелы вокруг **

    def root(self):
        """Get the root of numbers.

        Returns:
            float: num2-th root of num1.
        """
        return self.num1 ** (1 / self.num2)  # ИСПРАВЛЕНО: добавлены пробелы

    def remainer(self):
        """Get the remainder of numbers.

        Returns:
            float: The remainder of num1 divided by num2.
        """
        # Do I have to use the '.' period or full_stop in the numbers?
        return self.num1 % self.num2  # ИСПРАВЛЕНО: добавлены пробелы вокруг %

    def cube_root(self):
        """Get the cube root of numbers.

        Returns:
            float: The cube root of num1.
        """
        return self.num1 ** (1 / 3)  # ИСПРАВЛЕНО: добавлены пробелы

    def cube_exponent(self):
        """Get the cube exponent of numbers.

        Returns:
            float: num1 cubed.
        """
        return self.num1 ** 3  # ИСПРАВЛЕНО: добавлены пробелы вокруг **

    def square_root(self):
        """Get the square root of numbers.

        Returns:
            float: The square root of num1.
        """
        return self.num1 ** (1 / 2)  # ИСПРАВЛЕНО: добавлены пробелы

    def square_exponent(self):
        """Get the square exponent of numbers.

        Returns:
            float: num1 squared.
        """
        return self.num1 ** 2  # ИСПРАВЛЕНО: добавлены пробелы вокруг **

    def factorial(self):
        """Get the factorial of numbers."""
        pass

    def list_factors(self):
        """Get the list of factors of numbers."""
        pass

    def calculate_factorial(self):  # ИСПРАВЛЕНО: переименован метод
        """Calculate factorial of a number."""
        result = 1
        for i in range(1, self.num + 1):
            result = result * i  # ИСПРАВЛЕНО: добавлены пробелы вокруг *
        return result

    def lcm(self):  # ИСПРАВЛЕНО: переименован в нижний регистр
        """Get the LCM of numbers."""
        pass

    def hcf(self):  # ИСПРАВЛЕНО: переименован в нижний регистр
        """Get the HCF of numbers."""
        pass

    # class time: # Working with days calculator
    def age_calculator(self):
        """Get the age of the user."""
        # This is be very accurate and precise it should include
        # proper leap year and last birthday till now every detail.
        # Should show the preciseness in seconds when called.
        pass

    def days_calculator(self):
        """Get the days between two dates."""
        pass

    def leap_year(self):
        """Get the leap year of the user."""
        pass

    def perimeter(self):
        """Get the perimeter of the user."""
        pass

    class Trigonometry:
        """Class enriched with all the methods to solve
        basic trignometric problems
        """

        def pythagorean_theorem(self):
            """Get the pythagorean theorem of the user."""
            pass

        def find_hypotenuse(self):
            """Get the hypotenuse of the user."""
            pass

        def find_base(self):
            """Get the base of the user."""
            pass

        def find_perpendicular(self):
            """Get the perpendicular of the user."""
            pass

    # class Logarithms:
    # Learn more about Maths in general

    def quadratic_equation(self):
        """Get the quadratic equation of the user."""
        pass

    def open_system_calculator(self):
        """Open the calculator present on the machine
        of the user
        """
        # first identify the os
        # track the calculator
        # add a debugging feature like error handling
        # for linux and mac
        # if no such found then print a message to the user
        # that sorry dear it wasn't possible to so
        # then open it

    def take_inputs(self):
        """Take the inputs from the user in proper sucession."""
        while True:
            while True:
                try:
                    # self.num1 = float(input("Enter The First Number: "))
                    # self.num2 = float(input("Enter The Second Number: "))
                    pprint("Enter your number")
                    # validation check must be done
                    break
                except ValueError:
                    pprint("Please Enter A Valid Number")
                    continue
                # To let the user to know it is time to exit.
            pprint("Press 'q' to exit")
        # if self.num1 == "q" or self.num2 == "q":
        #     exit()  # Some how I need to exit it

    def greeting(self):
        """Greet the user with using Audio."""
        text_to_audio = "Welcome To The Calculator"
        self.gtts_object = gTTS(
            text=text_to_audio, lang="en", tld="co.in", slow=False
        )  # ИСПРАВЛЕНО: разбита длинная строка
        tts = self.gtts_object
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)  # Reset the BytesIO object to the beginning
        mixer.init()
        mixer.music.load(fp)
        mixer.music.play()
        while mixer.music.get_busy():
            time.Clock().tick(10)

    # Here OOP is not followed.
    def user_name(self):
        """Get the name of the user and have an option
        to greet him/her
        """
        self.name = input("Please enter your good name: ")
        # Making validation checks here
        text_to_audio = f"{self.name}"  # ИСПРАВЛЕНО: исправлена f-строка
        self.gtts_object = gTTS(
            text=text_to_audio, lang="en", tld="co.in", slow=False
        )  # ИСПРАВЛЕНО: разбита длинная строка
        tts = self.gtts_object
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)  # Reset the BytesIO object to the beginning
        mixer.init()
        mixer.music.load(fp)
        mixer.music.play()
        while mixer.music.get_busy():
            time.Clock().tick(10)

    def user_name_art(self):
        """Get the name of the user and have an option
        to show him his user name in art
        """
        # Default is to show = True, else False if user tries to disable it.

        # Tell him to show the time and date
        # print(art.text2art(self.name))
        # print(date and time of now)
        # Remove whitespaces from beginning and end
        # Remove middle name and last name
        # Remove special characters
        # Remove numbers
        f_name = self.name.split(" ")[0]
        f_name = f_name.strip()
        # Remove every number present in it
        # Will have to practice not logic
        f_name = "".join([i for i in f_name if not i.isdigit()])

        # perform string operations on it for the art to be displayed.
        # Remove white spaces
        # Remove middle name and last name
        # Remove special characters
        # Remove numbers
        # Remove everything

    class UnitConversion:
        """Class enriched with all the methods to convert units"""

        # Do we full-stops in generating documentations?

        def __init__(self):
            """Initialize the class with the required attributes."""
            self.take_inputs()

        def length(self):
            """Convert length units."""
            # It should have a meter to unit and unit to meter converter
            # Othe lengths units it should also have.
            # Like cm to pico meter and what not
            pass

        def area(self):
            """Calculate area of different shapes."""
            # This will to have multiple shapes and polygons to it
            # to improve it's area.
            # This will to have multiple shapes and polygons to it
            # to improve it's area.
            # I will try to use the best of the formula to do it
            # like the n number of polygons to be solved.
            pass

        def volume(self):
            """Calculate volume of different shapes."""
            # Different shapes and polygons to it to improve it's volume.
            pass

        def mass(self):
            """Convert mass units."""
            pass

        def time(self):
            """Convert time units."""
            pass

        def speed(self):
            """Convert speed units."""
            pass

        def temperature(self):
            """Convert temperature units."""
            pass

        def data(self):
            """Convert data units."""
            pass

        def pressure(self):
            """Convert pressure units."""
            pass

        def energy(self):
            """Convert energy units."""
            pass

        def power(self):
            """Convert power units."""
            pass

        def angle(self):
            """Convert angle units."""
            pass

        def force(self):
            """Convert force units."""
            pass

        def frequency(self):
            """Convert frequency units."""
            pass

        def take_inputs(self):
            """Take inputs for unit conversion."""
            pass

    class CurrencyConverter:
        """Class to handle currency conversions."""

        def __init__(self):
            self.take_inputs()

        def take_inputs(self):
            """Take inputs for currency conversion."""
            pass

        def convert(self):
            """Convert currencies."""
            pass

    class Commands:
        """Class to handle calculator commands."""

        def __init__(self):
            self.take_inputs()

        def previous_number(self):
            """Retrieve previous number."""
            pass

        def previous_operation(self):
            """Retrieve previous operation."""
            pass

        def previous_result(self):
            """Retrieve previous result."""
            pass

    def clear_screen(self):
        """Clear the console screen."""
        # Do I need a clear screen?
        # os.system("cls" if os.name == "nt" else "clear")
        # os.system("cls")
        # os.system("clear")
        pass


if __name__ == "__main__":
    operation_1 = Calculator()

    # Operations
    # User interaction
    # Study them properly and try to understand them.
    # Study them properly and try to understand them in very
    # detailed length. Please.
    # Add a function to continually ask for input until the user
    # enters a valid input.


# Let's explore colorma
# Also user log ins, and it saves user data and preferences.
# A feature of the least priority right now.

# List of features priority should be planned.


# Documentations are good to read and understand.
# A one stop solution is to stop and read the document.
# It is much better and easier to understand.