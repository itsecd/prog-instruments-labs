from test import *

if __name__ == "__main__":
    """
    Executes all test cases defined in the test module.

    This block of code runs the test suite using the unittest framework,
    allowing automatic discovery and execution of all test methods 
    defined in the imported test class.
    """
    try:
        unittest.main()
    except Exception as e:
        print(f"An error occurred while running tests: {e}")