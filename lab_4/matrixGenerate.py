import argparse

import numpy as np
from numpy.typing import NDArray

if __package__:
    from .matrix_io import write_matrix
else:
    from matrix_io import write_matrix


def writeMatrix(filePath: str, matrix: NDArray[np.generic], size: int) -> None:
    write_matrix(filePath, matrix, size)


def matrixGenerate(size: int) -> NDArray[np.int_]:
    return np.random.randint(-1000, 1000, (size, size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--size", type=int, default=2000, help="size of matrix to generate"
    )
    parser.add_argument(
        "-m1", "--matrix1-file", default="matrix1.txt", help="path to matrix1 file"
    )
    parser.add_argument(
        "-m2", "--matrix2-file", default="matrix2.txt", help="path to matrix2 file"
    )
    args = parser.parse_args()

    print(f"[RUN] => The path to file with first matrix is: {args.matrix1_file}")
    print(f"[RUN] => The path to file with second matrix is: {args.matrix2_file}")

    try:
        matrix1 = matrixGenerate(args.size)
        matrix2 = matrixGenerate(args.size)
        write_matrix(args.matrix1_file, matrix1, args.size)
        write_matrix(args.matrix2_file, matrix2, args.size)
    except Exception as error:
        print(f"[ERROR] => Error ro generate or write into a file matrix: {error}")


if __name__ == "__main__":
    main()
