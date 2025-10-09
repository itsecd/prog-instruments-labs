class Yahtzee:
    """
    class for calculating scores in Yahtzee game
    """

    @staticmethod
    def chance(dice):
        """
        This function calculating sum of all dice values
        :param dice: dice values
        :return:sum of all dice values
        """
        return sum(dice)

    @staticmethod
    def yahtzee(dice):
        """
        Yahtzee category - 50 points bonus for five of a kind
        :param dice: dice values
        :return: 50, if conditions to Yahtzee complete, else return 0
        """
        yahtzee_bonus = 50
        counts = Yahtzee.get_counts(dice)
        for i in range(len(counts)):
            if counts[i] == 5:
                return yahtzee_bonus
        return 0

    @staticmethod
    def score_by_number(dice, number):
        """
        Helper method to calculate score for specific number categories.
        :param dice: dice values
        :param number: number to score
        :return: sum of all dice that match the specified number
        """
        score_count = 0
        for i in dice:
            if i == number:
                score_count += number
        return score_count

    @staticmethod
    def ones(dice):
        """
        This function calculate sum of all dice showing the number 1.
        :param dice: dice values
        :return: score
        """
        return Yahtzee.score_by_number(dice, 1)

    @staticmethod
    def twos(dice):
        """
        This function calculate sum of all dice showing the number 2.
        :param dice: dice values
        :return: score
        """
        return Yahtzee.score_by_number(dice, 2)

    @staticmethod
    def threes(dice):
        """
        This function calculate sum of all dice showing the number 3.
        :param dice: dice values
        :return: score
        """
        return Yahtzee.score_by_number(dice, 3)

    @staticmethod
    def fours(dice):
        """
        This function calculate sum of all dice showing the number 4.
        :param dice: dice values
        :return: score
        """
        return Yahtzee.score_by_number(dice, 4)

    @staticmethod
    def fives(dice):
        """
        This function calculate sum of all dice showing the number 5.
        :param dice: dice values
        :return: score
        """
        return Yahtzee.score_by_number(dice, 5)

    @staticmethod
    def sixes(dice):
        """
        This function calculate sum of all dice showing the number 6.
        :param dice: dice values
        :return: score
        """
        return Yahtzee.score_by_number(dice, 6)

    @staticmethod
    def get_counts(dice):
        """
        This function counts frequency of each dice value.
        :param dice: dice values
        :return: Frequency counts where index represents dice value (1-6)
        """
        counts = [0] * 7
        for die in dice:
            if 1 <= die <= 6:
                counts[die] += 1
            else:
                raise ValueError("Incorrect die number in dice")
        return counts

    @staticmethod
    def score_pair(dice):
        """
        This function do sum of highest pair found
        :param dice: dice values
        :return: sum of the two dice in the highest scoring pair, or 0 if no pair exists
        """
        counts = Yahtzee.get_counts(dice)
        for value in range(len(counts) - 1, 0, -1):
            if counts[value] >= 2:
                return value * 2
        return 0

    @staticmethod
    def two_pair(dice):
        """
        This function do sum of two different pairs.
        :param dice: dice values
        :return: combined sum of two highest pairs, or 0 if fewer than two pairs exist
        """
        counts = Yahtzee.get_counts(dice)
        pairs = [i for i in range(1, 7) if counts[i] >= 2]
        if len(pairs) >= 2:
            return sum(pairs) * 2
        else:
            return 0

    @staticmethod
    def three_of_a_kind(dice):
        """
        This function do sum of three matching dice
        :param dice: dice values
        :return: sum of the three matching dice, or 0 if no three of a kind exists
        """
        counts = Yahtzee.get_counts(dice)
        for i in range(1, 7):
            if counts[i] >= 3:
                return i * 3
        return 0

    @staticmethod
    def four_of_a_kind(dice):
        """
        This function do sum of four matching dice
        :param dice: dice values
        :return: sum of the four matching dice, or 0 if no four of a kind exists
        """
        counts = Yahtzee.get_counts(dice)
        for i in range(1, 7):
            if counts[i] >= 4:
                return i * 4
        return 0

    @staticmethod
    def small_straight(dice):
        """
        This function do small Straight category - 15 points for sequence 1-2-3-4-5.
        :param dice: dice values
        :return: 15 if sequence 1-2-3-4-5 is present, otherwise 0
        """
        small_straight_bonus = 15
        counts = Yahtzee.get_counts(dice)
        required_numbers = [1, 2, 3, 4, 5]

        for number in required_numbers:
            if counts[number] == 0:
                return 0

        return small_straight_bonus

    @staticmethod
    def large_straight(dice):
        """
        This function do large Straight category - 20 points for sequence 2-3-4-5-6.
        :param dice: dice values
        :return: 20 if sequence 2-3-4-5-6 is present, otherwise 0
        """
        large_straight_bonus = 20
        counts = Yahtzee.get_counts(dice)
        required_numbers = [2, 3, 4, 5, 6]

        for number in required_numbers:
            if counts[number] == 0:
                return 0

        return large_straight_bonus

    @staticmethod
    def full_house(dice):
        """
        This function do Full House category - sum of all dice for combination of three of a kind and a pair
        :param dice: dice values
        :return: sum of all dice if full house condition is met, otherwise 0
        """
        counts = Yahtzee.get_counts(dice)

        has_three_of_a_kind = any(count == 3 for count in counts[1:])
        has_two_of_a_kind = any(count == 2 for count in counts[1:])

        if has_three_of_a_kind and has_two_of_a_kind:
            return sum(dice)
        return 0
