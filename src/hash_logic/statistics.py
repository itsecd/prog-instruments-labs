"""Module for running experiments and calculating statistics."""
import math
from .collision_finder import find_collision


def theoretical_attempts(bits: int) -> float:
    """
    Calculates theoretical expected attempts to find a collision (Birthday Paradox).

    Args:
        bits (int): The number of bits in the truncated hash.

    Returns:
        float: Expected number of attempts.

    Raises:
        ValueError: If bits parameter is less than or equal to zero.
        Exception: If any unexpected error occurs.
    """
    try:
        if bits <= 0:
            raise ValueError("Bits must be positive.")
        return math.sqrt(math.pi / 2.0 * (2 ** bits))
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise Exception(f"Error in theoretical_attempts: {e}")


def run_experiments(bits: int, count: int, settings: dict) -> dict:
    """
    Runs collision search multiple times to collect statistics.

    Args:
        bits (int): Number of bits for the hash.
        count (int): Number of experiments to run.
        settings (dict): Application settings dictionary.

    Returns:
        dict: Results including average attempts, theoretical attempts, and raw data.

    Raises:
        ValueError: If count is less than or equal to zero.
        Exception: If any unexpected error occurs.
    """
    try:
        if count <= 0:
            raise ValueError("Count must be positive.")

        total_attempts = 0
        collisions = []

        for i in range(count):
            print(f"\n--- Experiment {i + 1}/{count} ---")
            result = find_collision(bits, settings)
            total_attempts += result['attempts']
            collisions.append(result)

        avg_attempts = total_attempts / count
        theory = theoretical_attempts(bits)

        return {
            "average_attempts": avg_attempts,
            "theoretical_attempts": theory,
            "collisions": collisions
        }
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise Exception(f"Error in run_experiments: {e}")