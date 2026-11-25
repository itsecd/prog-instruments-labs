from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_matrix(path: str | Path) -> NDArray[np.generic]:
    with open(path, encoding="utf-8") as file:
        size = int(file.readline())
        rows = [list(map(int, file.readline().split())) for _ in range(size)]

    return np.array(rows)


def write_matrix(path: str | Path, matrix: NDArray[np.generic], size: int) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"{size}\n")
        for index, row in enumerate(matrix):
            if index:
                file.write("\n")
            file.write(" ".join(map(str, row)))
