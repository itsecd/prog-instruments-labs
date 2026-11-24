import numpy as np
from numpy.typing import NDArray


def read_matrix(file_path: str) -> NDArray[np.int64]:
    with open(file_path, encoding="utf-8") as file:
        size = int(file.readline())
        rows = [list(map(int, file.readline().split())) for _ in range(size)]

    return np.array(rows, dtype=np.int64)


def main() -> None:
    try:
        matrix1 = read_matrix("matrix1.txt")
        matrix2 = read_matrix("matrix2.txt")
        expected = read_matrix("matrixResult.txt")
        product = matrix1 @ matrix2

        if np.array_equal(product, expected):
            print("[RUN] => Matrixes are equal")
        else:
            print("[RUN] => Matrixes are not equal")
    except (OSError, ValueError) as error:
        print(f"[ERROR] => Fail to verify matrix: {error}")


if __name__ == "__main__":
    main()
