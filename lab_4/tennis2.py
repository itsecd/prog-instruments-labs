class TennisGame2:
    SCORE_NAMES = ["Love", "Fifteen", "Thirty", "Forty"]
    DEUCE = "Deuce"
    ADVANTAGE_P1 = "Advantage player1"
    ADVANTAGE_P2 = "Advantage player2"
    WIN_P1 = "Win for player1"
    WIN_P2 = "Win for player2"

    def __init__(self, player1_name, player2_name):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.p1points = 0
        self.p2points = 0

    def won_point(self, player_name):
        if player_name == "player1":
            self.p1_score()
        else:
            self.p2_score()

    def _get_score_name(self, points):
        if points < 4:
            return self.SCORE_NAMES[points]
        return ""

    def score(self):
        result = ""
        if self.p1points == self.p2points and self.p1points < 3:
            result = self._get_score_name(self.p1points) + "-All"
        if self.p1points == self.p2points and self.p1points > 2:
            result = self.DEUCE

        p1res = ""
        p2res = ""
        if self.p1points > 0 and self.p2points == 0:
            p1res = self._get_score_name(self.p1points)
            p2res = self._get_score_name(self.p2points)
            result = p1res + "-" + p2res
        if self.p2points > 0 and self.p1points == 0:
            p2res = self._get_score_name(self.p2points)
            p1res = self._get_score_name(self.p1points)
            result = p1res + "-" + p2res

        if self.p1points > self.p2points and self.p1points < 4:
            p1res = self._get_score_name(self.p1points)
            p2res = self._get_score_name(self.p2points)
            result = p1res + "-" + p2res
        if self.p2points > self.p1points and self.p2points < 4:
            p2res = self._get_score_name(self.p2points)
            p1res = self._get_score_name(self.p1points)
            result = p1res + "-" + p2res

        if self.p1points > self.p2points and self.p2points >= 3:
            result = self.ADVANTAGE_P1

        if self.p2points > self.p1points and self.p1points >= 3:
            result = self.ADVANTAGE_P2

        if (
            self.p1points >= 4
            and self.p2points >= 0
            and (self.p1points - self.p2points) >= 2
        ):
            result = self.WIN_P1
        if (
            self.p2points >= 4
            and self.p1points >= 0
            and (self.p2points - self.p1points) >= 2
        ):
            result = self.WIN_P2
        return result

    def set_p1_score(self, number):
        for i in range(number):
            self.p1_score()

    def set_p2_score(self, number):
        for i in range(number):
            self.p2_score()

    def p1_score(self):
        self.p1points += 1

    def p2_score(self):
        self.p2points += 1