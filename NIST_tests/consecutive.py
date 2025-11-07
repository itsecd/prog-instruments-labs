import math

def consecutive_bit_test(sequence:str)->float:
    """
    This funtion do A test for identical consecutive bits
    :param sequence: is dequence
    :return: p_value
    """
    length = len(sequence)
    if length!=0:
        one_counter = 0
        for i in sequence:
            if i == '1':
                one_counter+=1
            elif i=='0':
                one_counter+=0
            else:
                raise ValueError("Sequence is non binary")
        dzeta = one_counter / length

        if abs(dzeta - 0.5) >= 2 / math.sqrt(length):
            return 0

        v_n = 0

        for i in range(length - 1):
            if sequence[i] != sequence[i + 1]:
                v_n += 1

        p_value = math.erfc(abs(v_n - 2 * length * dzeta * (1 - dzeta)) /
                            (2 * math.sqrt(2 * length) * dzeta * (1 - dzeta)))
    else:
        raise ValueError("Sequence is empty")
    return p_value