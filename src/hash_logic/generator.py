"""Module for generating random data."""
import random
import string

def generate_random_string(length: int) -> str:
    """
    Generates a random alphanumeric string of a specified length.

    Args:
        length (int): Length of the string to generate.

    Returns:
        str: Generated random string.

    Raises:
        ValueError: If length is less than or equal to 0.
        Exception: If any unexpected error occurs.
    """
    try:
        if length <= 0:
            raise ValueError("Length must be a positive integer.")
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=length))
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise Exception(f"Unexpected error in generate_random_string: {e}")