import argparse
import random


def argument_parsing() -> list[str]:
    """
    Разделение аргументов, введённых пользователем в консоли
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', type=str, help='path to output file')
    args = parser.parse_args()
    return [args.output_file]


def generateDRV() -> int:
    """
    Генерация ДСВ (дискретной случайной величины) принимающей значения 0 и 1
    """
    return random.randint(0, 1)


def generateDRVec() -> list[int]:
    """
    Генерация ДСВек (дискретного случайного вектора) из ДСВ выше
    """
    res = list()
    for i in range(128):
        res += [generateDRV()]
    return res


def writeDRVec(output_file: str) -> None:
    """
    Создание файла и запись результата генерации в него
    """
    with open(output_file, '+w') as out_file:
        DRVec = generateDRVec()
        DRVec = map(str, DRVec)
        out_file.write(''.join(DRVec))


def main():
    output_file = argument_parsing()[0]
    writeDRVec(output_file)


if __name__ == '__main__':
    main()
