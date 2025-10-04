

class Yahtzee:

    @staticmethod
    def chance(die1, die2, die3, die4, die5):
        return die1 + die2 + die3 + die4 + die5

    @staticmethod
    def yahtzee(dice):
        for die in dice:
            if die != dice[0]:
                return 0
        return 50

    def __init__(self, die1, die2, die3, die4, die5):
        self.dice = [die1, die2, die3, die4, die5]

    @staticmethod
    def ones(die1, die2, die3, die4, die5):
        score = 0
        for die in [die1, die2, die3, die4, die5]:
            if die == 1:
                score += 1
        return score

    @staticmethod
    def twos(die1, die2, die3, die4, die5):
        score = 0
        for die in [die1, die2, die3, die4, die5]:
            if die == 2:
                score += 2
        return score

    @staticmethod
    def threes(die1, die2, die3, die4, die5):
        score = 0
        for die in [die1, die2, die3, die4, die5]:
            if die == 3:
                score += 3
        return score

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
    def score_pair(die1, die2, die3, die4, die5):
        counts = [0] * 6
        for die in [die1, die2, die3, die4, die5]:
            counts[die - 1] += 1

        for value in range(6, 0, -1):
            if counts[value - 1] >= 2:
                return value * 2
        return 0

    @staticmethod
    def two_pair(die1, die2, die3, die4, die5):
        counts = [0] * 6
        for die in [die1, die2, die3, die4, die5]:
            counts[die - 1] += 1

        pairs = []
        for i in range(5, -1, -1):
            if counts[i] == 2:
                pairs.append(i + 1)

        if len(pairs) == 2:
            return sum(pairs) * 2
        return 0

    @staticmethod
    def four_of_a_kind(die1, die2, die3, die4, die5):
        counts = [0] * 6
        for die in [die1, die2, die3, die4, die5]:
            counts[die - 1] += 1

        for value in range(1, 7):
            if counts[value - 1] >= 4:
                return value * 4
        return 0

    @staticmethod
    def three_of_a_kind(die1, die2, die3, die4, die5):
        counts = [0] * 6
        for die in [die1, die2, die3, die4, die5]:
            counts[die - 1] += 1

        for i in range(6):
            if counts[i] == 3:
                return (i + 1) * 3
        return 0

    @staticmethod
    def small_straight(die1, die2, die3, die4, die5):
        counts = [0] * 6
        for die in [die1, die2, die3, die4, die5]:
            counts[die - 1] += 1

        if all(counts[i] >= 1 for i in range(5)):
            return 15
        return 0

    @staticmethod
    def large_straight(die1, die2, die3, die4, die5):
        counts = [0] * 6
        for die in [die1, die2, die3, die4, die5]:
            counts[die - 1] += 1

        if all(counts[i] == 1 for i in range(1, 5)):
            return 20
        return 0

    @staticmethod
    def full_house(die1, die2, die3, die4, die5):
        counts = [0] * 6
        for die in [die1, die2, die3, die4, die5]:
            counts[die - 1] += 1

        has_pair = 2 in counts
        has_three = 3 in counts

        if has_pair and has_three:
            pair_value = counts.index(2) + 1
            three_value = counts.index(3) + 1
            return pair_value * 2 + three_value * 3
        return 0