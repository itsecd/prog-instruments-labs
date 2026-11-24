import numpy as num

def readMatrixFromFile (filePath: str) -> num.ndarray:
    """
    read matrix
    """
    with open(filePath, "r", encoding="utf-8") as file:
        size = int(file.readline().strip())

        matrix = []
        for row in range(size):
            rowSplit = file.readline().split()
            row = list(map(int, rowSplit))
            matrix.append(row)
    
    matrixNum = num.array(matrix)

    return matrixNum


def main():
    try:
        matrix1 = readMatrixFromFile("matrix1.txt")
        matrix2 = readMatrixFromFile("matrix2.txt")
        matrixCpp = readMatrixFromFile("matrixResult.txt")

        matrixPython = num.dot(matrix1, matrix2)

        if num.array_equal(matrixPython, matrixCpp):
            print("[RUN] => Matrixes are equal")
            return

        print("[RUN] => Matrixes are`t equal")


    except Exception as e:
        print(f"[ERROR] => Fail to read matrix: {e}")

if __name__ == "__main__" :
    main()