from enum import Enum


class YatzyCategory(Enum):
    CHANCE = 0
    YATZY = 1
    ONES = 2
    TWOS = 3
    THREES = 4
    FOURS = 5
    FIVES = 6
    SIXES = 7
    PAIR = 8
    THREE_OF_A_KIND = 9
    FOUR_OF_A_KIND = 10
    SMALL_STRAIGHT = 11
    LARGE_STRAIGHT = 12
    TWO_PAIRS = 13
    FULL_HOUSE = 14


DICE_VALUES = [6, 5, 4, 3, 2, 1]


class Yatzy:

    def get_frequencies(self, dice):
        """Подсчитывает, сколько раз встречается каждое значение"""
        freq = {i: 0 for i in DICE_VALUES}
        for die in dice:
            freq[die] += 1
        return freq

    def sum_of_number(self, dice_frequencies, number):
        """Возвращает сумму всех костей с указанным числом"""
        return dice_frequencies[number] * number

    def of_a_kind(self, dice_frequencies, count):
        """Возвращает сумму очков для N одинаковых костей (наибольшего номинала)"""
        for i in DICE_VALUES:
            if dice_frequencies[i] >= count:
                return i * count
        return 0

    def pair(self, dice_frequencies):
        """Находит одну пару"""
        return self._of_a_kind(dice_frequencies, 2)

    def two_pairs(self, dice_frequencies):
        """Находит две пары."""
        pairs = [i for i in DICE_VALUES if dice_frequencies[i] >= 2]
        return sum(i * 2 for i in pairs) if len(pairs) == 2 else 0

    def yatzy(self, dice_frequencies):
        """Проверяет (все пять одинаковые)."""
        return 50 if 5 in dice_frequencies.values() else 0

    def straight(self, dice, expected):
        """Проверяет стрит"""
        return sum(dice) if sorted(dice) == expected else 0

    def full_house(self, dice_frequencies, dice):
        """Проверяет фулл-хаус"""
        if 2 in dice_frequencies.values() and 3 in dice_frequencies.values():
            return sum(dice)
        return 0
    
    def score(self, dice, category: YatzyCategory) -> int:
        freq = self.get_frequencies(dice)
        
        match category:
            case YatzyCategory.CHANCE:
                return sum(dice)

            case YatzyCategory.YATZY:
                return self.yatzy(freq)

            case YatzyCategory.ONES:
                return self.sum_of_number(freq, 1)

            case YatzyCategory.TWOS:
                return self.sum_of_number(freq, 2)

            case YatzyCategory.THREES:
                return self.sum_of_number(freq, 3)

            case YatzyCategory.FOURS:
                return self.sum_of_number(freq, 4)

            case YatzyCategory.FIVES:
                return self.sum_of_number(freq, 5)

            case YatzyCategory.SIXES:
                return self.sum_of_number(freq, 6)

            case YatzyCategory.PAIR:
                return self.pair(freq)

            case YatzyCategory.THREE_OF_A_KIND:
                return self.of_a_kind(freq, 3)

            case YatzyCategory.FOUR_OF_A_KIND:
                return self.of_a_kind(freq, 4)

            case YatzyCategory.SMALL_STRAIGHT:
                return self.straight(dice, [1, 2, 3, 4, 5])

            case YatzyCategory.LARGE_STRAIGHT:
                return self.straight(dice, [2, 3, 4, 5, 6])

            case YatzyCategory.TWO_PAIRS:
                return self.two_pairs(freq)

            case YatzyCategory.FULL_HOUSE:
                return self.full_house(freq, dice)

        return 0
