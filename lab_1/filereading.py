def read_file(path: str) -> str:
    '''
    Чтение строки из файла
    '''
    with open(path, 'r+') as file:
        return file.readline()[:-1]


def read_vector(line: str) -> list[bool]:
    '''
    Чтение вектора из строки
    '''
    result = []
    for symb in line:
        if symb == '0':
            result += [False]
        elif symb == '1':
            result += [True]
        else:
            raise ValueError(
                'Uncorrect file format! Symbol: ' + symb + 'is used!')
    return result


def read_sequence(path: str) -> list[int]:
    '''
    Чтение вектора из файла
    '''
    return read_vector(read_file(path))
