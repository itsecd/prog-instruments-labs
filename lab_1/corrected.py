# This is like making a package.lock file for npm package.
# Yes, I should be making it.
__author__ = "Nitkarsh Chourasia"
__version__ = "0.0.0"  # SemVer
__license__ = "MIT"

from gtts import gTTS
from pygame import mixer, time
from io import BytesIO
from pprint import pprint


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
        return self.num1 / self.num2

    def power(self):
        """Get the power of numbers.

        Returns:
            float: num1 raised to the power of num2.
        """
        return self.num1 ** self.num2

    def root(self):
        """Get the root of numbers.

        Returns:
            float: num2-th root of num1.
        """
        return self.num1 ** (1 / self.num2)

    def remainer(self):
        """Get the remainder of numbers.

        Returns:
            float: The remainder of num1 divided by num2.
        """
        return self.num1 % self.num2

    def cube_root(self):
        """Get the cube root of numbers.

        Returns:
            float: The cube root of num1.
        """
        return self.num1 ** (1 / 3)

    def cube_exponent(self):
        """Get the cube exponent of numbers.

        Returns:
            float: num1 cubed.
        """
        return self.num1 ** 3

    def square_root(self):
        """Get the square root of numbers.

        Returns:
            float: The square root of num1.
        """
        return self.num1 ** (1 / 2)

    def square_exponent(self):
        """Get the square exponent of numbers.

        Returns:
            float: num1 squared.
        """
        return self.num1 ** 2

    def factorial(self):
        """Get the factorial of numbers."""
        pass

    def list_factors(self):
        """Get the list of factors of numbers."""
        pass

    def calculate_factorial(self):
        """Calculate factorial of a number."""
        result = 1
        for i in range(1, self.num + 1):
            result = result * i
        return result

    def lcm(self):
        """Get the LCM of numbers."""
        pass

    def hcf(self):
        """Get the HCF of numbers."""
        pass

    def age_calculator(self):
        """Get the age of the user."""
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
        basic trigonometric problems.
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

    def quadratic_equation(self):
        """Get the quadratic equation of the user."""
        pass

    def open_system_calculator(self):
        """Open the calculator present on the machine of the user."""
        pass

    def take_inputs(self):
        """Take the inputs from the user in proper succession."""
        while True:
            while True:
                try:
                    pprint("Enter your number")
                    break
                except ValueError:
                    pprint("Please Enter A Valid Number")
                    continue
            pprint("Press 'q' to exit")

    def greeting(self):
        """Greet the user with using Audio."""
        text_to_audio = "Welcome To The Calculator"
        self.gtts_object = gTTS(
            text=text_to_audio, lang="en", tld="co.in", slow=False
        )
        tts = self.gtts_object
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        mixer.init()
        mixer.music.load(fp)
        mixer.music.play()
        while mixer.music.get_busy():
            time.Clock().tick(10)

    def user_name(self):
        """Get the name of the user and have an option to greet him/her."""
        self.name = input("Please enter your good name: ")
        text_to_audio = f"{self.name}"
        self.gtts_object = gTTS(
            text=text_to_audio, lang="en", tld="co.in", slow=False
        )
        tts = self.gtts_object
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        mixer.init()
        mixer.music.load(fp)
        mixer.music.play()
        while mixer.music.get_busy():
            time.Clock().tick(10)

    def user_name_art(self):
        """Get the name of the user and show it as art."""
        f_name = self.name.split(" ")[0]
        f_name = f_name.strip()
        f_name = "".join([i for i in f_name if not i.isdigit()])

    class UnitConversion:
        """Class enriched with all the methods to convert units."""

        def __init__(self):
            self.take_inputs()

        def length(self):
            """Convert length units."""
            pass

        def area(self):
            """Calculate area of different shapes."""
            pass

        def volume(self):
            """Calculate volume of different shapes."""
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
        pass


if __name__ == "__main__":
    operation_1 = Calculator()