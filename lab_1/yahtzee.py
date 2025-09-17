

class Yahtzee:

    @staticmethod
    def chance(die1, die2, die3, die4, die5):
        sum_of_dice = 0
        sum_of_dice += die1
        sum_of_dice += die2
        sum_of_dice += die3
        sum_of_dice += die4
        sum_of_dice += die5
        return sum_of_dice

    @staticmethod
    def yahtzee(dice):
        counts = [0] * (len(dice) + 1)
        for die in dice:
            counts[die - 1] += 1
        for i in range(len(counts)):
            if counts[i] == 5:
                return 50
        return 0

    def __init__(self, die1, die2, die3, die4, die5):
        self.dice = [0] * 5
        self.dice[0] = die1
        self.dice[1] = die2
        self.dice[2] = die3
        self.dice[3] = die4
        self.dice[4] = die5

    @staticmethod
    def ones(die1, die2, die3, die4, die5):
        score = 0
        if (die1 == 1):
            score += 1
        if (die2 == 1):
            score += 1
        if (die3 == 1):
            score += 1
        if (die4 == 1):
            score += 1
        if (die5 == 1):
            score += 1
        return score

    @staticmethod
    def twos(die1, die2, die3, die4, die5):
        score = 0
        if (die1 == 2):
            score += 2
        if (die2 == 2):
            score += 2
        if (die3 == 2):
            score += 2
        if (die4 == 2):
            score += 2
        if (die5 == 2):
            score += 2
        return score

    @staticmethod
    def threes(die1, die2, die3, die4, die5):
        score = 0
        if (die1 == 3):
            score += 3
        if (die2 == 3):
            score += 3
        if (die3 == 3):
            score += 3
        if (die4 == 3):
            score += 3
        if (die5 == 3):
            score += 3
        return score

    def fours(self):
        score = 0
        for i in range(5):
            if (self.dice[i] == 4):
                score += 4
        return score

    def fives(self):
        score = 0
        i = 0
        for i in range(len(self.dice)): 
            if (self.dice[i] == 5):
                score = score + 5
        return score

    def sixes(self):
        score = 0
        for i in range(len(self.dice)):
            if (self.dice[i] == 6):
                score = score + 6
        return score
    
    @staticmethod
    def score_pair(die1, die2, die3, die4, die5):
        counts = [0] * 6
        counts[die1 - 1] += 1
        counts[die2 - 1] += 1
        counts[die3 - 1] += 1
        counts[die4 - 1] += 1
        counts[die5 - 1] += 1
        at = 0
        for i in range(6):
            if (counts[6 - i - 1] == 2):
                return (6 - i) * 2
        return 0
    
    @staticmethod
    def two_pair(die1, die2, die3, die4, die5):
        counts = [0] * 6
        counts[die1 - 1] += 1
        counts[die2 - 1] += 1
        counts[die3 - 1] += 1
        counts[die4 - 1] += 1
        counts[die5 - 1] += 1
        pair_count = 0
        pair_score = 0
        for i in range(6):
            if (counts[6 - i - 1] == 2):
                pair_count = pair_count + 1
                pair_score += (6 - i)
                    
        if (pair_count == 2):
            return pair_score * 2
        else:
            return 0
    
    @staticmethod
    def four_of_a_kind(die1, die2, die3, die4, die5):
        counts = [0] * 6
        counts[die1 - 1] += 1
        counts[die2 - 1] += 1
        counts[die3 - 1] += 1
        counts[die4 - 1] += 1
        counts[die5 - 1] += 1
        for i in range(6):
            if (counts[i] == 4):
                return (i + 1) * 4
        return 0

    @staticmethod
    def three_of_a_kind(die1, die2, die3, die4, die5):
        counts = [0] * 6
        counts[die1 - 1] += 1
        counts[die2 - 1] += 1
        counts[die3 - 1] += 1
        counts[die4 - 1] += 1
        counts[die5 - 1] += 1
        for i in range(6):
            if (counts[i] == 3):
                return (i + 1) * 3
        return 0

    @staticmethod
    def small_straight(die1, die2, die3, die4, die5):
        counts = [0] * 6
        counts[die1 - 1] += 1
        counts[die2 - 1] += 1
        counts[die3 - 1] += 1
        counts[die4 - 1] += 1
        counts[die5 - 1] += 1
        if (counts[0] == 1 and
            counts[1] == 1 and
            counts[2] == 1 and
            counts[3] == 1 and
            counts[4] == 1):
            return 15
        return 0

    @staticmethod
    def large_straight(die1, die2, die3, die4, die5):
        counts = [0] * 6
        counts[die1 - 1] += 1
        counts[die2 - 1] += 1
        counts[die3 - 1] += 1
        counts[die4 - 1] += 1
        counts[die5 - 1] += 1
        if (counts[1] == 1 and
            counts[2] == 1 and
            counts[3] == 1 and
            counts[4] == 1 and
            counts[5] == 1):
            return 20
        return 0

    @staticmethod
    def full_house(die1, die2, die3, die4, die5):
        counts = []
        has_pair = False
        i = 0
        pair_value = 0
        has_three_of_kind = False
        three_of_kind_value = 0

        counts = [0] * 6
        counts[die1 - 1] += 1
        counts[die2 - 1] += 1
        counts[die3 - 1] += 1
        counts[die4 - 1] += 1
        counts[die5 - 1] += 1

        for i in range(6):
            if (counts[i] == 2):
                has_pair = True
                pair_value = i + 1

        for i in range(6):
            if (counts[i] == 3):
                has_three_of_kind = True
                three_of_kind_value = i + 1

        if (has_pair and has_three_of_kind):
            return pair_value * 2 + three_of_kind_value * 3
        else:
            return 0
