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
        sum = 0
        if (d1 == 1):
            sum += 1
        if (d2 == 1):
            sum += 1
        if (d3 == 1):
            sum += 1
        if (d4 == 1):
            sum += 1
        if (d5 == 1): 
            sum += 1

        return sum
    

    @staticmethod
    def twos(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for twos category."""
        sum = 0
        if (d1 == 2):
             sum += 2
        if (d2 == 2):
             sum += 2
        if (d3 == 2):
             sum += 2
        if (d4 == 2):
             sum += 2
        if (d5 == 2):
             sum += 2
        return sum
    
    @staticmethod
    def threes(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for threes category."""
        s = 0
        if (d1 == 3):
             s += 3
        if (d2 == 3):
             s += 3
        if (d3 == 3):
             s += 3
        if (d4 == 3):
             s += 3
        if (d5 == 3):
             s += 3
        return s
    

    def __init__(d1: int, d2: int, d3: int, d4: int, _5: int) -> int:
        """Initialize Yahtzee game with five dice values."""
        self.dice = [0]*5
        self.dice[0] = d1
        self.dice[1] = d2
        self.dice[2] = d3
        self.dice[3] = d4
        self.dice[4] = _5
    
    def fours(self) -> int:
        """Calculate score for fours category."""
        sum = 0
        for at in range(5):
            if (self.dice[at] == 4): 
                sum += 4
        return sum
    

    def fives(self) -> int:
        """Calculate score for fives category."""
        s = 0
        i = 0
        for i in range(len(self.dice)): 
            if (self.dice[i] == 5):
                s = s + 5
        return s
    

    def sixes(self) -> int:
        """Calculate score for sixes category."""
        sum = 0
        for at in range(len(self.dice)): 
            if (self.dice[at] == 6):
                sum = sum + 6
        return sum
    
    @staticmethod
    def score_pair(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for the highest pair."""
        counts = [0]*6
        counts[d1-1] += 1
        counts[d2-1] += 1
        counts[d3-1] += 1
        counts[d4-1] += 1
        counts[d5-1] += 1
        at = 0
        for at in range(6):
            if (counts[6-at-1] == 2):
                return (6-at)*2
        return 0
    
    @staticmethod
    def two_pair(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score if there are two pairs."""
        counts = [0]*6
        counts[d1-1] += 1
        counts[d2-1] += 1
        counts[d3-1] += 1
        counts[d4-1] += 1
        counts[d5-1] += 1
        n = 0
        score = 0
        for i in range(6):
            if (counts[6-i-1] == 2):
                n = n+1
                score += (6-i)
                    
        if (n == 2):
            return score * 2
        else:
            return 0
    
    @staticmethod
    def four_of_a_kind(_1: int, _2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for four of a kind."""
        tallies = [0]*6
        tallies[_1-1] += 1
        tallies[_2-1] += 1
        tallies[d3-1] += 1
        tallies[d4-1] += 1
        tallies[d5-1] += 1
        for i in range(6):
            if (tallies[i] == 4):
                return (i+1) * 4
        return 0
    

    @staticmethod
    def three_of_a_kind(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for three of a kind."""
        t = [0]*6
        t[d1-1] += 1
        t[d2-1] += 1
        t[d3-1] += 1
        t[d4-1] += 1
        t[d5-1] += 1
        for i in range(6):
            if (t[i] == 3):
                return (i+1) * 3
        return 0
    

    @staticmethod
    def smallStraight(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Check if dice show small straight (1-2-3-4-5) and return score."""
        tallies = [0]*6
        tallies[d1-1] += 1
        tallies[d2-1] += 1
        tallies[d3-1] += 1
        tallies[d4-1] += 1
        tallies[d5-1] += 1
        if (tallies[0] == 1 and
            tallies[1] == 1 and
            tallies[2] == 1 and
            tallies[3] == 1 and
            tallies[4] == 1):
            return 15
        return 0
    

    @staticmethod
    def largeStraight(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Check if dice show large straight (2-3-4-5-6) and return score."""
        tallies = [0]*6
        tallies[d1-1] += 1
        tallies[d2-1] += 1
        tallies[d3-1] += 1
        tallies[d4-1] += 1
        tallies[d5-1] += 1
        if (tallies[1] == 1 and
            tallies[2] == 1 and
            tallies[3] == 1 and
            tallies[4] == 1
            and tallies[5] == 1):
            return 20
        return 0
    

    @staticmethod
    def fullHouse(d1: int, d2: int, d3: int, d4: int, d5: int) -> int:
        """Calculate score for full house."""
        tallies = []
        _2 = False
        i = 0
        _2_at = 0
        _3 = False
        _3_at = 0

        tallies = [0]*6
        tallies[d1-1] += 1
        tallies[d2-1] += 1
        tallies[d3-1] += 1
        tallies[d4-1] += 1
        tallies[d5-1] += 1

        for i in range(6):
            if (tallies[i] == 2): 
                _2 = True
                _2_at = i+1
            

        for i in range(6):
            if (tallies[i] == 3): 
                _3 = True
                _3_at = i+1
            

        if (_2 and _3):
            return _2_at * 2 + _3_at * 3
        else:
            return 0
