import argparse

import numpy as np
from numpy.typing import NDArray


def write_matrix(file_path: str, matrix: NDArray[np.int64], size: int) -> None:
    result = [" ".join(map(str, row)) for row in matrix]

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"{size}\n")
        file.write("\n".join(result))


def generate_matrix(size: int) -> NDArray[np.int64]:
    return np.random.randint(-1000, 1000, (size, size), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=2000, help="size of matrix to generate")
    parser.add_argument("-m1", "--matrix1-file", default="matrix1.txt", help="path to matrix1 file")
    parser.add_argument("-m2", "--matrix2-file", default="matrix2.txt", help="path to matrix2 file")
    args = parser.parse_args()

    print(f"[RUN] => The path to file with first matrix is: {args.matrix1_file}")
    print(f"[RUN] => The path to file with second matrix is: {args.matrix2_file}")

    try:
        matrix1 = generate_matrix(args.size)
        matrix2 = generate_matrix(args.size)
        write_matrix(args.matrix1_file, matrix1, args.size)
        write_matrix(args.matrix2_file, matrix2, args.size)
    except (OSError, ValueError) as error:
        print(f"[ERROR] => Could not generate or write matrices: {error}")


if __name__ == "__main__":
    main()
