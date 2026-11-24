import argparse
import numpy as num

def writeMatrix (filePath: str, matrix: num.ndarray, size:int) -> None:
    """
    write matrix in a file
    """
    result = []

    for row in matrix:
        rowString = map(str, row)
        result.append(" ".join(rowString))
    
    resultMatrix = "\n".join(result)

    with open(filePath, "w", encoding="utf-8") as file:
        file.write(str(size) + "\n")
        file.write(resultMatrix)


def matrixGenerate (size: int) -> num.ndarray:
    """
    generate matrix
    """
    matrix = num.random.randint(-1000, 1000, (size,size))
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", '--size', type=int, default=2000, help='size of matrix to generate')
    parser.add_argument("-m1", '--matrix1-file', type=str, default='matrix1.txt', help='path to matrix1 file')
    parser.add_argument("-m2", '--matrix2-file', type=str, default='matrix2.txt', help='path to matrix2 file')
    args = parser.parse_args()

    print(f"[RUN] => The path to file with first matrix is: {args.matrix1_file}")
    print(f"[RUN] => The path to file with second matrix is: {args.matrix2_file}")

    try:
        matrix1 = matrixGenerate(args.size)
        matrix2 = matrixGenerate(args.size)

        writeMatrix(args.matrix1_file, matrix1, args.size)
        writeMatrix(args.matrix2_file, matrix2, args.size)

    except Exception as e:
        print(f"[ERROR] => Error ro generate or write into a file matrix: {e}")

if __name__ == "__main__" :
    main()