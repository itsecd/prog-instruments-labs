"""Module for finding hash collisions."""
from tqdm import tqdm
from .generator import generate_random_string
from .hasher import get_truncated_hash


def find_collision(bits: int, settings: dict) -> dict:
    """
    Finds a hash collision for a given bit length using brute force.

    Args:
        bits (int): Number of bits for the hash.
        settings (dict): Application settings dictionary.

    Returns:
        dict: A dictionary containing collision details ('str1', 'str2', 'hash', 'attempts').

    Raises:
        ValueError: If bits parameter is invalid.
        Exception: If any unexpected error occurs during the search.
    """
    try:
        allowed_bits = settings["hasher"]["allowed_bits"]
        str_length = settings["generator"]["default_length"]

        if bits not in allowed_bits:
            raise ValueError(f"Bits must be one of {allowed_bits}.")

        seen_hashes = {}
        attempts = 0

        with tqdm(desc=f"Searching {bits}-bit collision", unit=" attempts") as pbar:
            while True:
                attempts += 1
                pbar.update(1)

                candidate = generate_random_string(str_length)
                h = get_truncated_hash(candidate, bits, allowed_bits)

                if h in seen_hashes and seen_hashes[h] != candidate:
                    return {
                        "str1": seen_hashes[h],
                        "str2": candidate,
                        "hash": h,
                        "attempts": attempts
                    }
                seen_hashes[h] = candidate
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise Exception(f"Unexpected error in find_collision: {e}")