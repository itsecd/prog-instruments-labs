class TennisGame2:
    # 1. Выносим магические строки в константы
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

    def score(self):
        result = ""
        if self.p1points == self.p2points and self.p1points < 3:
            if self.p1points == 0:
                result = self.SCORE_NAMES[0]
            if self.p1points == 1:
                result = self.SCORE_NAMES[1]
            if self.p1points == 2:
                result = self.SCORE_NAMES[2]
            result += "-All"
        if self.p1points == self.p2points and self.p1points > 2:
            result = self.DEUCE

        p1res = ""
        p2res = ""
        if self.p1points > 0 and self.p2points == 0:
            if self.p1points == 1:
                p1res = self.SCORE_NAMES[1]
            if self.p1points == 2:
                p1res = self.SCORE_NAMES[2]
            if self.p1points == 3:
                p1res = self.SCORE_NAMES[3]

            p2res = self.SCORE_NAMES[0]
            result = p1res + "-" + p2res
        if self.p2points > 0 and self.p1points == 0:
            if self.p2points == 1:
                p2res = self.SCORE_NAMES[1]
            if self.p2points == 2:
                p2res = self.SCORE_NAMES[2]
            if self.p2points == 3:
                p2res = self.SCORE_NAMES[3]

            p1res = self.SCORE_NAMES[0]
            result = p1res + "-" + p2res

        if self.p1points > self.p2points and self.p1points < 4:
            if self.p1points == 2:
                p1res = self.SCORE_NAMES[2]
            if self.p1points == 3:
                p1res = self.SCORE_NAMES[3]
            if self.p2points == 1:
                p2res = self.SCORE_NAMES[1]
            if self.p2points == 2:
                p2res = self.SCORE_NAMES[2]
            result = p1res + "-" + p2res
        if self.p2points > self.p1points and self.p2points < 4:
            if self.p2points == 2:
                p2res = self.SCORE_NAMES[2]
            if self.p2points == 3:
                p2res = self.SCORE_NAMES[3]
            if self.p1points == 1:
                p1res = self.SCORE_NAMES[1]
            if self.p1points == 2:
                p1res = self.SCORE_NAMES[2]
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