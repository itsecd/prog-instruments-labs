import math

def freq_test(sequence:str)->float:
    """
    This function do Frequency bitwise test.
    :param sequence:is sequence
    :return:p_value
    """
    one_counter = 0
    zero_counter = 0
    length = len(sequence)
    if length !=0:
        for i in sequence:
            if i == '1':
                one_counter+=1
            elif i == '0':
                zero_counter+=1
            else:
                raise ValueError("Sequence is non binary")
    else:
        raise ValueError("Sequence is empty")
    s_n = (one_counter - zero_counter) / math.sqrt(length)
    p_value = math.erfc(abs(s_n) / math.sqrt(2))
    return p_value