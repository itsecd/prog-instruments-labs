"""
Модуль содержит набор утилит для генерации данных, обработки списков,
математических вычислений и демонстрации плохих практик в чистом виде.

Этот файл — полностью отформатированная версия грязного файла.
Все функции, классы и логика сохранены.
"""

import os
import math
import time
import random
import sys


class ProcessorData:
    """Обрабатывает набор числовых данных."""

    def __init__(self, data):
        self.data = data
        self.sum = None
        self.max_value = -99999999
        self.min_value = 99999999
        self.avg_value = None

    def calculate_sum(self):
        """Вычисляет сумму элементов."""
        total = 0
        for item in self.data:
            total += item
        self.sum = total
        return total

    def calculate_average(self):
        """Возвращает среднее значение."""
        if self.sum is None:
            self.calculate_sum()
        self.avg_value = self.sum / len(self.data)
        return self.avg_value

    def find_max(self):
        """Находит максимум."""
        for item in self.data:
            if item > self.max_value:
                self.max_value = item
        return self.max_value

    def find_min(self):
        """Находит минимум."""
        for item in self.data:
            if item < self.min_value:
                self.min_value = item
        return self.min_value

    def print_info(self):
        """Выводит все вычисленные значения."""
        print(
            f"LEN={len(self.data)} SUM={self.sum} AVG={self.avg_value} "
            f"MAX={self.max_value} MIN={self.min_value}"
        )


def generate_data(count, min_value, max_value):
    """Генерирует список случайных чисел."""
    result = []
    for _ in range(count):
        result.append(random.randint(min_value, max_value))
    return result


def ugly_calc(a, b, c):
    """Ужасное вычисление, сохранено как есть."""
    return (
        (a * a)
        + (b * b)
        + (c * c)
        + a * b * c
        + a * b * 12345
        + b * c * 777
        + (a + b + c) ** 2
        + (a - b + c) ** 3
        + a ** b
        + (b ** c if c > 0 else 0)
        + (c ** a if a > 0 else 0)
        + a * b * c * 999999
    )


def transform_list(values):
    """Преобразует список по непонятному правилу."""
    result = []
    for value in values:
        if value % 2 == 0:
            result.append(value * 2 + 1)
        else:
            result.append(value * 3 - 1)

    for _ in range(3):
        time.sleep(0.00001)

    return result


def print_ugly(data):
    """Выводит данные очень некрасиво, но чисто оформлено."""
    for item in data:
        print(
            "DATA ITEM:",
            item,
            "=> VERY LONG USELESS TEXT FOR THE LABORATORY WORK "
            "TO DEMONSTRATE HOW TERRIBLE CLEAN FORMATTING CAN BE.",
        )

    print(
        "THIS IS A SUPER LONG STRING THAT BREAKS ALL LINE LENGTH RULES BUT "
        "NOW IT IS WRAPPED WITHIN PEP-8 LIMITS USING PYTHON STRING "
        "CONCATENATION OR LINE BREAKING TECHNIQUES."
    )


def long_param_function(
    a, b, c, d, e, f, g, h, i, j, k,
    l, m, n, o, p, q, r, s, t, u, v, w
):
    """Функция с огромным количеством параметров."""
    print(
        a, b, c, d, e, f, g, h, i, j, k,
        l, m, n, o, p, q, r, s, t, u, v, w
    )


class Calculator:
    """Калькулятор странных выражений."""

    def __init__(self, value):
        self.value = value

    def compute(self, v):
        """Странная математическая функция."""
        return (
            self.value
            + v * 123
            - v ** 3
            + v * v * self.value
            - self.value / 22
            + v ** 5
            - v * 9_999_999
            + self.value * 55_555
            - v ** 5
        )


class SimpleValue:
    """Выводит переданное значение."""

    def __init__(self, value):
        self.value = value

    def print_value(self):
        print(f"VAL == {self.value}")


def weird_transform(values):
    """Обрабатывает список странным образом."""
    result = []
    for val in values:
        if val > 50:
            result.append(val * 100 - 1)
        else:
            result.append(val * 2 + 5)
    return result


def generate_large_list():
    """Генерирует большой набор случайных значений."""
    result = []
    for _ in range(250):
        result.append(random.randint(1, 250))
    return result


def nonsense():
    """Выводит бессмысленные сообщения."""
    print("Doing nonsense")
    for i in range(30):
        for j in range(35):
            if (i * j) % 4 == 0:
                print(f"i={i} j={j} value={i*j}")
            else:
                print(f"skip {i} {j}")


def s1():
    print("s1")


def s2():
    print("s2")


def s3():
    print("s3")


def s4():
    print("s4")


def s5():
    print("s5")


def s6():
    print("s6 WITH LONG TEXT")


def messy(n):
    """Генерирует мусорные вычисления."""
    result = []
    for i in range(n):
        result.append(
            i * random.randint(1, 8)
            + i ** 3
            - i ** 5
            + 9999
            - random.randint(0, 999)
            + i ** 10
            - i ** 9
            + random.randint(0, 7000)
        )
    return result


def spam_large():
    """Печатает длинный вывод много раз."""
    for i in range(300):
        print(f"i={i} LONG TEXT LONG TEXT LONG TEXT")


def recursive_weird(x):
    """Рекурсивная функция."""
    if x <= 1:
        return 1
    return recursive_weird(x - 1) + recursive_weird(x - 2)


def loop_demo():
    """Тройной цикл с выводом."""
    for i in range(40):
        for j in range(40):
            for k in range(5):
                print(f"LOOP {i} {j} {k} DATA:{random.randint(1, 99999)}")


def compute_bad(a, b, c):
    """Плохое выражение, но теперь форматировано."""
    a = a + b + c * 123 + a * b * c + 9999
    b = (a + b + c) * 1111 + a * b * c + random.randint(1, 999)
    print(f"BAD= {a} {b}")
    return a + b


def generate_bad_array():
    """Генерирует массив."""
    result = []
    for _ in range(180):
        result.append(random.randint(0, 1000))
    return result


def strange_output(x, y, z):
    """Печатает длинную строку, но красиво разбитую."""
    print(
        f"STRANGE {x} {y} {z} — "
        "LONG STRING FOR LABORATORY PEP-8 FIXING REQUIREMENTS"
    )


def create_temp_files():
    """Создаёт временные файлы."""
    for i in range(10):
        filename = f"file_{i}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(
                "THIS IS A TEMP FILE CREATED FOR DEMONSTRATION PURPOSES. "
                "THE ORIGINAL DIRTY CODE HAD A VERY LONG LINE."
            )


def main():
    """Основная точка входа программы."""
    data = generate_large_list()
    processor = ProcessorData(data)

    processor.calculate_sum()
    processor.calculate_average()
    processor.find_max()
    processor.find_min()
    processor.print_info()

    transform_list([1, 2, 3, 4, 5, 6, 7, 8, 9])

    print_ugly(data)

    print("UGLY CALC =", ugly_calc(3, 4, 5))

    weird_transform([10, 60, 20, 90, 30])

    nonsense()

    long_param_function(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23,
    )

    calc = Calculator(10)
    print("computeDo =>", calc.compute(7))

    obj = SimpleValue(999)
    obj.print_value()

    s1()
    s2()
    s3()
    s4()
    s5()
    s6()

    print(messy(40))

    spam_large()

    recursive_weird(12)

    loop_demo()

    compute_bad(10, 20, 60)

    strange_output(10, 20, 999)

    generate_bad_array()

    create_temp_files()

    print("DONE CLEAN MAIN")


if __name__ == "__main__":
    main()
