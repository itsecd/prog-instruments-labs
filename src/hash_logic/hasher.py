"""Module for computing and truncating hashes."""
import hashlib


def get_truncated_hash(data: str, bits: int, allowed_bits: list) -> str:
    """
    Computes SHA-256 hash and truncates it to the specified bit length.

    Args:
        data (str): The input string to hash.
        bits (int): Number of bits to keep.
        allowed_bits (list): List of allowed bit lengths.

    Returns:
        str: Truncated hash as a hexadecimal string.

    Raises:
        ValueError: If bits is not in the allowed list.
        TypeError: If data is not a string.
        Exception: If any unexpected error occurs.
    """
    try:
        if bits not in allowed_bits:
            raise ValueError(f"Bits must be one of: {allowed_bits}.")
        if not isinstance(data, str):
            raise TypeError("Data must be a string.")

        hex_chars = bits // 4
        hashed = hashlib.sha256(data.encode('utf-8')).hexdigest()
        return hashed[:hex_chars]
    except (ValueError, TypeError) as ve:
        raise ve
    except Exception as e:
        raise Exception(f"Unexpected error in get_truncated_hash: {e}")