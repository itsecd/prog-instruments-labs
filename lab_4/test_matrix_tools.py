import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from matrinxMultiVerification import verify_product
from matrix_io import read_matrix, write_matrix


class MatrixToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_round_trip_preserves_file_format(self) -> None:
        path = self.directory / "matrix.txt"
        matrix = np.array([[2, -3], [0, 5]])

        write_matrix(path, matrix, 2)

        self.assertEqual(path.read_text(encoding="utf-8"), "2\n2 -3\n0 5")
        np.testing.assert_array_equal(read_matrix(path), matrix)

    def test_verify_product(self) -> None:
        first = self.directory / "first.txt"
        second = self.directory / "second.txt"
        result = self.directory / "result.txt"
        write_matrix(first, np.array([[1, 2], [3, 4]]), 2)
        write_matrix(second, np.array([[0, -1], [1, 2]]), 2)

        write_matrix(result, np.array([[2, 3], [4, 5]]), 2)
        self.assertTrue(verify_product(first, second, result))

        write_matrix(result, np.array([[2, 3], [5, 4]]), 2)
        self.assertFalse(verify_product(first, second, result))

    def test_missing_input_propagates_to_caller(self) -> None:
        missing = self.directory / "missing.txt"
        with self.assertRaises(FileNotFoundError):
            verify_product(missing, missing, missing)


if __name__ == "__main__":
    unittest.main()
