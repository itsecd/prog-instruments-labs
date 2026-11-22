import pytest
from seq_analyze import (
    psi, psi_sum, freq_p_value, zeta, V_N,
    repeat_p_value, analyze_block, V, seq_hi_squared
)

# --- ПРИМИТИВНЫЕ ТЕСТЫ ---


def test_psi():
    # Проверка функции psi (возвращает 1 для True, -1 для False)
    assert psi(True) == 1
    assert psi(False) == -1


def test_zeta():
    # zeta вычисляет долю единиц в последовательности
    seq = [True, False, True, True, False]
    # 3 единицы из 5
    assert zeta(seq) == 0.6


def test_analyze_block():
    # Тест на самую длинную последовательность из единиц
    assert analyze_block([True, True, False, True, True, True]) == 3
    assert analyze_block([False, False, False]) == 0
    assert analyze_block([True, True, True, True]) == 4


def test_V():
    # Проверка группировки по 8 элементов (блокам)
    seq = [True] * 8 + [True, True] + [False] * 6
    # Блок 1: [True]*8 -> bl=8 (res_V[3] += 1)
    # Блок 2: [True, True, False]*2 -> bl=2 (res_V[1] += 1)
    res = V(seq)
    assert res[1] == 1
    assert res[3] == 1

# --- ПРОДВИНУТЫЙ ТЕСТ №1: ПАРАМЕТРИЗАЦИЯ ---
# Согласно лекции, тестируем нормальное поведение и граничные значения.


@pytest.mark.parametrize(
    "seq, expected_vn",
    [
        # Нормальное поведение (каждый раз меняется)
        ([True, False, True, False, True], 4),
        # Граничное условие (нет изменений)
        ([True, True, True, True], 0),
        ([False, True], 1),                    # Минимальная длина
        ([True, False, True], 2),              # Чередование
    ]
)
def test_V_N(seq, expected_vn):
    # V_N считает количество смен знака (знакоперемен)
    assert V_N(seq) == expected_vn


def test_psi_sum():
    # Проверка вспомогательной суммы S_N
    # Для [True, False, True, False] -> 1 + (-1) + 1 + (-1) = 0
    seq = [True, False, True, False]
    assert psi_sum(seq) == 0.0
