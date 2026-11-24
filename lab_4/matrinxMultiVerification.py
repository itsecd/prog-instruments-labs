import numpy as np
from numpy.typing import NDArray

from matrix_io import read_matrix


def readMatrixFromFile(filePath: str) -> NDArray[np.generic]:
    return read_matrix(filePath)


def main() -> None:
    try:
        matrix1 = read_matrix("matrix1.txt")
        matrix2 = read_matrix("matrix2.txt")
        matrix_cpp = read_matrix("matrixResult.txt")
        matrix_python = np.dot(matrix1, matrix2)

        if np.array_equal(matrix_python, matrix_cpp):
            print("[RUN] => Matrixes are equal")
            return

        print("[RUN] => Matrixes are`t equal")
    except Exception as error:
        print(f"[ERROR] => Fail to read matrix: {error}")


if __name__ == "__main__":
    main()
