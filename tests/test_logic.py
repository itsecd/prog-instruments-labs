"""Module for unit testing core logic."""
import unittest
from src.hash_logic.generator import generate_random_string
from src.hash_logic.hasher import get_truncated_hash
from src.hash_logic.statistics import theoretical_attempts


class TestHashLogic(unittest.TestCase):
    """Unit tests for the hash logic modules."""

    def setUp(self):
        """Setup mock settings for tests.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        self.mock_allowed_bits = [8, 12, 16]

    def test_generate_random_string(self):
        """Tests if the generator respects length constraints.

        Args:
            None

        Returns:
            None

        Raises:
            AssertionError: If the generated string length does not match the expected length,
                or if ValueError is not raised for negative lengths.
            Exception: If any unexpected error occurs during the test execution.
        """
        try:
            res = generate_random_string(15)
            self.assertEqual(len(res), 15)
            with self.assertRaises(ValueError):
                generate_random_string(-5)
        except Exception as e:
            self.fail(f"Test failed with unexpected error: {e}")

    def test_get_truncated_hash(self):
        """Tests hash truncation sizes and invalid inputs.

        Args:
            None

        Returns:
            None

        Raises:
            AssertionError: If the truncated hash lengths do not match the expected byte sizes,
                or if ValueError is not raised for invalid bit sizes.
            Exception: If any unexpected error occurs during the test execution.
        """
        try:
            hash8 = get_truncated_hash("test", 8, self.mock_allowed_bits)
            hash12 = get_truncated_hash("test", 12, self.mock_allowed_bits)
            hash16 = get_truncated_hash("test", 16, self.mock_allowed_bits)

            self.assertEqual(len(hash8), 2)
            self.assertEqual(len(hash12), 3)
            self.assertEqual(len(hash16), 4)

            with self.assertRaises(ValueError):
                get_truncated_hash("test", 10, self.mock_allowed_bits)
        except Exception as e:
            self.fail(f"Test failed with unexpected error: {e}")

    def test_theoretical_attempts(self):
        """Tests calculation of theoretical expectations.

        Args:
            None

        Returns:
            None

        Raises:
            AssertionError: If the calculated theoretical attempts fall outside the expected bounds.
            Exception: If any unexpected error occurs during the test execution.
        """
        try:
            theory8 = theoretical_attempts(8)
            self.assertTrue(19 < theory8 < 21)
        except Exception as e:
            self.fail(f"Test failed with unexpected error: {e}")


if __name__ == "__main__":
    unittest.main()