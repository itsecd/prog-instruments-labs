from scipy.special import gammainc
from math import erfc, sqrt


def psi(omega: bool) -> int:
    '''
    Случайная величина как функция от координаты
    '''
    if omega:
        return 1
    else:
        return -1


def psi_sum(seq: list[bool]) -> float:
    '''
    Вспомогательная сумма S_N
    '''
    res = 0
    for u in seq:
        res += psi(u)
    return abs(res)/sqrt(len(seq))


def freq_p_value(seq: list[bool]) -> float:
    '''
    Результат частотного побитового теста
    '''
    return erfc(psi_sum(seq)/sqrt(2))


def zeta(seq: list[bool]) -> float:
    '''
    Вспомогательная сумма зета
    '''
    res = 0
    for x in seq:
        res += int(x)
    return res/len(seq)


def V_N(seq: list[bool]) -> float:
    '''
    Вспомогательная сумма V_N
    '''
    res = int(0)
    for i in range(len(seq)-1):
        if seq[i] != seq[i+1]:
            res += 1
    return res


def repeat_p_value(seq: list[bool]) -> float:
    '''
    Результат теста на число знакоперемен
    '''
    zta = zeta(seq)
    if (abs(zta-0.5) >= (2/sqrt(len(seq)))):
        return 0
    Vn = V_N(seq)
    return erfc(abs(Vn-2*len(seq)*zta*(1-zta))/(2*sqrt(2*len(seq))*zta*(1-zta)))


# Тест на самую длинную последовательность из единиц в блоках
def analyze_block(block: list[bool]) -> int:
    '''
    Обработка одного блока
    '''
    res = 0
    i = 0
    while (i < len(block)):
        tmp = 0
        while (i < len(block) and block[i]):
            tmp += 1
            i += 1
        if (tmp > res):
            res = tmp
        i += 1
    return res


def V(seq: list[bool]) -> list[int]:
    '''
    Сбор статистики по блокам
    '''
    res_V = [0, 0, 0, 0]
    for i in range(0, len(seq), 8):
        bl = analyze_block(seq[i:i+8])
        if bl <= 1:
            res_V[0] += 1
        elif bl == 2:
            res_V[1] += 1
        elif bl == 3:
            res_V[2] += 1
        else:
            res_V[3] += 1
    return res_V


def seq_hi_squared(seq: list[bool]) -> float:
    '''
    Подсчет величины Хи-квадрат
    '''
    pi = [0.2148, 0.3672, 0.2305, 0.1875]

    V_vect = V(seq)

    hi_squared = float(0)
    for i in range(4):
        tmp = (V_vect[i]-16*pi[i])
        hi_squared += tmp*tmp/(16*pi[i])
    return hi_squared


def max_lenght_p_value(seq: list[bool]) -> float:
    '''
    Подсчет резульатат теста 3
    '''
    return 1 - gammainc(1.5, seq_hi_squared(seq)/2)


def get_statistics(seq: list[bool]) -> list[float]:
    return [freq_p_value(seq), repeat_p_value(seq), max_lenght_p_value(seq)]


def print_statistics(datalist: list[float], language: str) -> None:
    '''
    Печать итоговой статистики на экран
    '''
    print('_____________________________')
    print('language = ', language)
    freq = datalist[0]
    print('frequency analysis p_value = ', freq)
    rep = datalist[1]
    print('repeating analysis p_value = ', rep)
    max_len = datalist[2]
    print('maximum block lenght p_value = ', max_len)
    if (freq >= 0.01 and rep >= 0.01 and max_len >= 0.01):
        print('All tested PASSED, vector is random')
    else:
        print('One or more test NOT PASSED, vector isn\'nt random')
    print('_____________________________')


def write_statistics(datalist: list[float], language: str) -> None:
    '''
    Запись итоговой статистики в файл
    '''
    with open('statistics.txt', 'a+') as out_file:
        out_file.write('_____________________________\n')
        out_file.write('language = ' + language+'\n')
        freq = datalist[0]
        out_file.write('frequency analysis p_value = ' + str(freq)+'\n')
        rep = datalist[1]
        out_file.write('repeating analysis p_value = ' + str(rep)+'\n')
        max_len = datalist[2]
        out_file.write('maximum block lenght p_value = '+str(max_len)+'\n')
        if (freq >= 0.01 and rep >= 0.01 and max_len >= 0.01):
            out_file.write('All tested PASSED, vector is random\n')
        else:
            out_file.write(
                'One or more test NOT PASSED, vector isn\'nt random\n')
        out_file.write('_____________________________\n')


def analyze_sequence(seq: list[bool], language: str) -> None:
    stats = get_statistics(seq)
    print_statistics(stats, language)
    write_statistics(stats, language)
