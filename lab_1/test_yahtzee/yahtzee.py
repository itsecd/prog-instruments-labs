class Yahtzee:
    DICE_COUNT = 5
    MAX_DIE_VALUE = 6

    @staticmethod
    def chance(dice1, dice2, dice3, dice4, dice5):
        return dice1 + dice2 + dice3 + dice4 + dice5

    @staticmethod
    def yahtzee(dice):
        first_die = dice[0]
        for die in dice:
            if die != first_die:
                return 0
        return 50

    @staticmethod
    def ones(dice1, dice2, dice3, dice4, dice5):
        score = 0
        if dice1 == 1:
            score += 1
        if dice2 == 1:
            score += 1
        if dice3 == 1:
            score += 1
        if dice4 == 1:
            score += 1
        if dice5 == 1:
            score += 1
        return score

    @staticmethod
    def twos(dice1, dice2, dice3, dice4, dice5):
        score = 0
        if dice1 == 2:
            score += 2
        if dice2 == 2:
            score += 2
        if dice3 == 2:
            score += 2
        if dice4 == 2:
            score += 2
        if dice5 == 2:
            score += 2
        return score

    @staticmethod
    def threes(dice1, dice2, dice3, dice4, dice5):
        score = 0
        if dice1 == 3:
            score += 3
        if dice2 == 3:
            score += 3
        if dice3 == 3:
            score += 3
        if dice4 == 3:
            score += 3
        if dice5 == 3:
            score += 3
        return score

    def __init__(self, dice1, dice2, dice3, dice4, dice5):
        self.dice = [dice1, dice2, dice3, dice4, dice5]

    def fours(self):
        score = 0
        for die in self.dice:
            if die == 4:
                score += 4
        return score

    def fives(self):
        score = 0
        for die in self.dice:
            if die == 5:
                score += 5
        return score

    def sixes(self):
        score = 0
        for die in self.dice:
            if die == 6:
                score += 6
        return score

    @staticmethod
    def _count_dice_values(dice1, dice2, dice3, dice4, dice5):
        counts = [0] * Yahtzee.MAX_DIE_VALUE
        counts[dice1 - 1] += 1
        counts[dice2 - 1] += 1
        counts[dice3 - 1] += 1
        counts[dice4 - 1] += 1
        counts[dice5 - 1] += 1
        return counts

    @staticmethod
    def score_pair(dice1, dice2, dice3, dice4, dice5):
        counts = Yahtzee._count_dice_values(dice1, dice2, dice3, dice4, dice5)
        for i in range(Yahtzee.MAX_DIE_VALUE - 1, -1, -1):
            if counts[i] >= 2:
                return (i + 1) * 2
        return 0

    @staticmethod
    def two_pair(dice1, dice2, dice3, dice4, dice5):
        counts = Yahtzee._count_dice_values(dice1, dice2, dice3, dice4, dice5)
        pairs = []
        for i in range(Yahtzee.MAX_DIE_VALUE - 1, -1, -1):
            if counts[i] >= 2:
                pairs.append(i + 1)

        if len(pairs) == 2:
            return sum(pairs) * 2
        return 0

    @staticmethod
    def four_of_a_kind(dice1, dice2, dice3, dice4, dice5):
        counts = Yahtzee._count_dice_values(dice1, dice2, dice3, dice4, dice5)
        for i in range(Yahtzee.MAX_DIE_VALUE):
            if counts[i] >= 4:
                return (i + 1) * 4
        return 0

    @staticmethod
    def three_of_a_kind(dice1, dice2, dice3, dice4, dice5):
        counts = Yahtzee._count_dice_values(dice1, dice2, dice3, dice4, dice5)
        for i in range(Yahtzee.MAX_DIE_VALUE):
            if counts[i] >= 3:
                return (i + 1) * 3
        return 0

    @staticmethod
    def small_straight(dice1, dice2, dice3, dice4, dice5):
        unique_values = {dice1, dice2, dice3, dice4, dice5}
        return 15 if unique_values == {1, 2, 3, 4, 5} else 0

    @staticmethod
    def large_straight(dice1, dice2, dice3, dice4, dice5):
        unique_values = {dice1, dice2, dice3, dice4, dice5}
        return 20 if unique_values == {2, 3, 4, 5, 6} else 0

    @staticmethod
    def full_house(dice1, dice2, dice3, dice4, dice5):
        counts = Yahtzee._count_dice_values(dice1, dice2, dice3, dice4, dice5)
        has_three = False
        has_two = False

        for count in counts:
            if count == 3:
                has_three = True
            elif count == 2:
                has_two = True

        if has_three and has_two:
            three_value = 0
            two_value = 0
            for i in range(Yahtzee.MAX_DIE_VALUE):
                if counts[i] == 3:
                    three_value = i + 1
                elif counts[i] == 2:
                    two_value = i + 1
            return three_value * 3 + two_value * 2

        return 0