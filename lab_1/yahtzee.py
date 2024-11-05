class Yahtzee:
    @staticmethod
    def chance(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate chance score, which is the sum of all dice."""
        return d1 + d2 + d3 + d4 + d5

    @staticmethod
    def yahtzee(dice: List[int]) -> int:
        """Return Yahtzee score of 50 if all dice have the same number, else 0."""
        counts = [0] * (len(dice) + 1)
        for die in dice:
            counts[die - 1] += 1
        return 50 if 5 in counts else 0

    @staticmethod
    def ones(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for ones category."""
        total = 0
        if d1 == 1:
            total += 1
        if d2 == 1:
            total += 1
        if d3 == 1:
            total += 1
        if d4 == 1:
            total += 1
        if d5 == 1:
            total += 1
        return total

    @staticmethod
    def twos(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for twos category."""
        total = 0
        if d1 == 2:
            total += 2
        if d2 == 2:
            total += 2
        if d3 == 2:
            total += 2
        if d4 == 2:
            total += 2
        if d5 == 2:
            total += 2
        return total

    @staticmethod
    def threes(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for threes category."""
        total = 0
        if d1 == 3:
            total += 3
        if d2 == 3:
            total += 3
        if d3 == 3:
            total += 3
        if d4 == 3:
            total += 3
        if d5 == 3:
            total += 3
        return total

    def __init__(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Initialize Yahtzee game with five dice values."""
        self.dice = [0] * 5
        self.dice[0] = d1
        self.dice[1] = d2
        self.dice[2] = d3
        self.dice[3] = d4
        self.dice[4] = d5

    def fours(self) -> int:
        """Calculate score for fours category."""
        total = 0
        for die in range(5):
            if self.dice[die] == 4:
                total += 4
        return total

    def fives(self) -> int:
        """Calculate score for fives category."""
        total = 0
        for die in range(len(self.dice)):
            if self.dice[die] == 5:
                total = total + 5
        return total

    def sixes(self) -> int:
        """Calculate score for sixes category."""
        total = 0
        for die in range(len(self.dice)):
            if self.dice[die] == 6:
                total = total + 6
        return total

    @staticmethod
    def score_pair(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for the highest pair."""
        counts = [0] * 6
        counts[d1 - 1] += 1
        counts[d2 - 1] += 1
        counts[d3 - 1] += 1
        counts[d4 - 1] += 1
        counts[d5 - 1] += 1
        for die in range(6):
            if counts[6 - die - 1] == 2:
                return (6 - die) * 2
        return 0

    @staticmethod
    def two_pair(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score if there are two pairs."""
        counts = [0] * 6
        counts[d1 - 1] += 1
        counts[d2 - 1] += 1
        counts[d3 - 1] += 1
        counts[d4 - 1] += 1
        counts[d5 - 1] += 1
        pairs = 0
        score = 0
        for i in range(6):
            if counts[6 - i - 1] == 2:
                pairs = pairs + 1
                score += 6 - i

        if pairs == 2:
            return score * 2
        else:
            return 0

    @staticmethod
    def four_of_a_kind(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for four of a kind."""
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
    def three_of_a_kind(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for three of a kind."""
        tallies = [0] * 6
        tallies[d1 - 1] += 1
        tallies[d2 - 1] += 1
        tallies[d3 - 1] += 1
        tallies[d4 - 1] += 1
        tallies[d5 - 1] += 1
        for i in range(6):
            if tallies[i] == 3:
                return (i + 1) * 3
        return 0

    @staticmethod
    def smallStraight(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Check if dice show small straight (1-2-3-4-5) and return score."""
        tallies = [0] * 6
        tallies[d1 - 1] += 1
        tallies[d2 - 1] += 1
        tallies[d3 - 1] += 1
        tallies[d4 - 1] += 1
        tallies[d5 - 1] += 1
        if (
            tallies[0] == 1
            and tallies[1] == 1
            and tallies[2] == 1
            and tallies[3] == 1
            and tallies[4] == 1
        ):
            return 15
        return 0

    @staticmethod
    def largeStraight(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Check if dice show large straight (2-3-4-5-6) and return score."""
        tallies = [0] * 6
        tallies[d1 - 1] += 1
        tallies[d2 - 1] += 1
        tallies[d3 - 1] += 1
        tallies[d4 - 1] += 1
        tallies[d5 - 1] += 1
        if (
            tallies[1] == 1
            and tallies[2] == 1
            and tallies[3] == 1
            and tallies[4] == 1
            and tallies[5] == 1
        ):
            return 20
        return 0

    @staticmethod
    def full_house(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for full house."""
        tallies = [0] * 6
        has_two_of_a_kind = False
        two_of_a_kind_value = 0
        has_three_of_a_kind = False
        three_of_a_kind_value = 0

        tallies[d1 - 1] += 1
        tallies[d2 - 1] += 1
        tallies[d3 - 1] += 1
        tallies[d4 - 1] += 1
        tallies[d5 - 1] += 1

        for i in range(6):
            if tallies[i] == 2:
                has_two_of_a_kind = True
                two_of_a_kind_value = i + 1

        for i in range(6):
            if tallies[i] == 3:
                has_three_of_a_kind = True
                three_of_a_kind_value = i + 1

        if has_two_of_a_kind and has_three_of_a_kind:
            return two_of_a_kind_value * 2 + three_of_a_kind_value * 3
        return 0
