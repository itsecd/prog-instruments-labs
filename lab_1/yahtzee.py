class Yahtzee:

    @staticmethod
    def chance(d1, d2, d3, d4, d5):
        return d1 + d2 + d3 + d4 + d5

    @staticmethod
    def yahtzee(dice):
        counts = [0] * (len(dice) + 1)
        for die in dice:
            counts[die - 1] += 1
        for count in counts:
            if count == 5:
                return 50
        return 0

    @staticmethod
    def ones(d1, d2, d3, d4, d5):
        score = 0
        if d1 == 1:
            score += 1
        if d2 == 1:
            score += 1
        if d3 == 1:
            score += 1
        if d4 == 1:
            score += 1
        if d5 == 1:
            score += 1
        return score

    @staticmethod
    def twos(d1, d2, d3, d4, d5):
        score = 0
        if d1 == 2:
            score += 2
        if d2 == 2:
            score += 2
        if d3 == 2:
            score += 2
        if d4 == 2:
            score += 2
        if d5 == 2:
            score += 2
        return score

    @staticmethod
    def threes(d1, d2, d3, d4, d5):
        score = 0
        if d1 == 3:
            score += 3
        if d2 == 3:
            score += 3
        if d3 == 3:
            score += 3
        if d4 == 3:
            score += 3
        if d5 == 3:
            score += 3
        return score

    def __init__(self, d1, d2, d3, d4, d5):
        self.dice = [d1, d2, d3, d4, d5]

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
    def score_pair(d1, d2, d3, d4, d5):
        counts = [0] * 6
        counts[d1 - 1] += 1
        counts[d2 - 1] += 1
        counts[d3 - 1] += 1
        counts[d4 - 1] += 1
        counts[d5 - 1] += 1
        for value in range(6, 0, -1):
            if counts[value - 1] == 2:
                return value * 2
        return 0

    @staticmethod
    def two_pair(d1, d2, d3, d4, d5):
        counts = [0] * 6
        counts[d1 - 1] += 1
        counts[d2 - 1] += 1
        counts[d3 - 1] += 1
        counts[d4 - 1] += 1
        counts[d5 - 1] += 1
        pairs_count = 0
        total_score = 0
        for value in range(6, 0, -1):
            if counts[value - 1] == 2:
                pairs_count += 1
                total_score += value

        if pairs_count == 2:
            return total_score * 2
        else:
            return 0

    @staticmethod
    def four_of_a_kind(_1, _2, d3, d4, d5):
        tallies = [0] * 6
        tallies[d1 - 1] += 1
        tallies[d2 - 1] += 1
        tallies[d3 - 1] += 1
        tallies[d4 - 1] += 1
        tallies[d5 - 1] += 1
        for i in range(6):
            if tallies[i] == 4:
                return (i + 1) * 4
        return 0

    @staticmethod
    def three_of_a_kind(d1, d2, d3, d4, d5):
        counts = [0] * 6
        counts[d1 - 1] += 1
        counts[d2 - 1] += 1
        counts[d3 - 1] += 1
        counts[d4 - 1] += 1
        counts[d5 - 1] += 1
        for i in range(6):
            if counts[i] == 3:
                return (i + 1) * 3
        return 0

    @staticmethod
    def small_straight(d1, d2, d3, d4, d5):
        tallies = [0] * 6
        tallies[d1 - 1] += 1
        tallies[d2 - 1] += 1
        tallies[d3 - 1] += 1
        tallies[d4 - 1] += 1
        tallies[d5 - 1] += 1
        if (tallies[0] == 1 and tallies[1] == 1 and
                tallies[2] == 1 and tallies[3] == 1 and tallies[4] == 1):
            return 15
        return 0

    @staticmethod
    def large_straight(d1, d2, d3, d4, d5):
        tallies = [0] * 6
        tallies[d1 - 1] += 1
        tallies[d2 - 1] += 1
        tallies[d3 - 1] += 1
        tallies[d4 - 1] += 1
        tallies[d5 - 1] += 1
        if (tallies[1] == 1 and tallies[2] == 1 and
                tallies[3] == 1 and tallies[4] == 1 and tallies[5] == 1):
            return 20
        return 0

    @staticmethod
    def full_house(d1, d2, d3, d4, d5):
        has_pair = False
        pair_value = 0
        has_three = False
        three_value = 0

        tallies = [0] * 6
        tallies[d1 - 1] += 1
        tallies[d2 - 1] += 1
        tallies[d3 - 1] += 1
        tallies[d4 - 1] += 1
        tallies[d5 - 1] += 1

        for i in range(6):
            if tallies[i] == 2:
                has_pair = True
                pair_value = i + 1

        for i in range(6):
            if tallies[i] == 3:
                has_three = True
                three_value = i + 1

        if has_pair and has_three:
            return pair_value * 2 + three_value * 3
        else:
            return 0