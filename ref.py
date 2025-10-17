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
        freq = self.get_frequencies(dise)
        
        match category:
            case YatzyCategory.CHANCE:
                return sum(dice)

            case YatzyCategory.YATZY:
                return self.yatzy(freq)

            case YatzyCategory.ONES:
                # sum all the ones
                result = dice_frequencies[1]

            case YatzyCategory.TWOS:
                # sum all the twos
                result = dice_frequencies[2] * 2

            case YatzyCategory.THREES:
                # sum all the threes
                result = dice_frequencies[3] * 3

            case YatzyCategory.FOURS:
                # sum all the fours
                result = dice_frequencies[4] * 4

            case YatzyCategory.FIVES:
                # sum all the fives
                result = dice_frequencies[5] * 5

            case YatzyCategory.SIXES:
                # sum all the sixes
                result = dice_frequencies[6] * 6

            case YatzyCategory.PAIR:

                # score pair if two dice are the same
                pair_result = 0
                # score highest pair if there is more than one
                for i in DICE_VALUES:
                    if dice_frequencies[i] >= 2:
                        pair_result = i * 2
                        break

                result = pair_result

            case YatzyCategory.THREE_OF_A_KIND:

                # score if three dice are the same
                three_kind_result = 0
                for i in DICE_VALUES:
                    if dice_frequencies[i] >= 3:
                        three_kind_result = i * 3

                result = three_kind_result

            case YatzyCategory.FOUR_OF_A_KIND:

                # score if four dice are the same
                four_kind_result = 0
                for i in DICE_VALUES:
                    if dice_frequencies[i] >= 4:
                        four_kind_result = i * 4

                result = four_kind_result

            case YatzyCategory.SMALL_STRAIGHT:

                # score if dice contains 1,2,3,4,5
                small_straight_result = 0

                count = 0
                for frequency in dice_frequencies.values():
                    if frequency == 1:
                        count += 1

                if count == 5 and dice_frequencies[6] == 0:
                    for die in dice:
                        small_straight_result += die

                result = small_straight_result

            case YatzyCategory.LARGE_STRAIGHT:

                # score if dice contains 2,3,4,5,6
                large_straight_result = 0
                straight_count = 0
                for frequency in dice_frequencies.values():
                    if frequency == 1:
                        straight_count += 1

                if straight_count == 5 and dice_frequencies[1] == 0:
                    for die in dice:
                        large_straight_result += die

                result = large_straight_result

            case YatzyCategory.TWO_PAIRS:

                # score if there are two pairs
                two_pair_result = 0
                pair_count = 0
                for frequency in dice_frequencies.values():
                    if frequency >= 2:
                        pair_count += 1

                if pair_count == 2:
                    for i in DICE_VALUES:
                        if dice_frequencies[i] >= 2:
                            two_pair_result += i * 2

                result = two_pair_result

            case YatzyCategory.FULL_HOUSE:

                # score if there is a pair as well as three of a kind
                full_house_result = 0
                if 2 in dice_frequencies.values() and 3 in dice_frequencies.values():
                    for die in dice:
                        full_house_result += die

                result = full_house_result

        return result
