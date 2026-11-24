from pathlib import Path

import numpy as np
from numpy.typing import NDArray

if __package__:
    from .matrix_io import read_matrix
else:
    from matrix_io import read_matrix


def readMatrixFromFile(filePath: str) -> NDArray[np.generic]:
    return read_matrix(filePath)


def verify_product(
    first_path: str | Path, second_path: str | Path, result_path: str | Path
) -> bool:
    first = read_matrix(first_path)
    second = read_matrix(second_path)
    expected = read_matrix(result_path)
    return np.array_equal(np.dot(first, second), expected)


def main() -> None:
    try:
        if verify_product("matrix1.txt", "matrix2.txt", "matrixResult.txt"):
            print("[RUN] => Matrixes are equal")
            return

        print("[RUN] => Matrixes are`t equal")
    except Exception as error:
        print(f"[ERROR] => Fail to read matrix: {error}")


if __name__ == "__main__":
    main()
