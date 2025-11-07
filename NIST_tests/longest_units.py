from scipy import special

def longest_units_test(sequence:str,pi_values:list)->float:
    """
    this function do A test for identical consecutive bits.
    :param sequence:is sequence
    :param pi_values: pi values from metodichka
    :return: p_value
    """
    length = len(sequence)
    m = 8

    if length == 0:
        raise ValueError("Sequence must not be empty")
    v = [0, 0, 0, 0]

    for i in range(0, len(sequence), m):
        block = sequence[i:i + m]
        max_length = current = 0

        for bit in block:
            current = current + 1 if bit == '1' else 0
            max_length = max(max_length, current)

        match max_length:
            case max_length if max_length <= 1:
                v[0] += 1
            case 2:
                v[1] += 1
            case 3:
                v[2] += 1
            case max_length if max_length >= 4:
                v[3] += 1

    xi_square = sum(((v[i] - 16 * pi_values[i]) ** 2) / (16 * pi_values[i]) for i in range(len(v)))
    p_value = special.gammainc((3 / 2), (xi_square / 2))
    return p_value