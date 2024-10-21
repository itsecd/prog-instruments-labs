class TennisGameDefactored1:

    def __init__(self, player1_name, player2_name):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.p1_points = 0
        self.p2_points = 0

    def won_point(self, player_name):
        if player_name == self.player1_name:
            self.p1_points += 1
        else:
            self.p2_points += 1

    def score(self):
        result = ""
        temp_score = 0
        if (self.p1_points == self.p2_points):
            result = {
                0: "Love-All",
                1: "Fifteen-All",
                2: "Thirty-All",
                3: "Forty-All",
            }.get(self.p1_points, "Deuce")
        elif (self.p1_points >= 4 or self.p2_points >= 4):
            minus_result = self.p1_points-self.p2_points
            if (minus_result == 1):
                result = "Advantage " + self.player1_name
            elif (minus_result == -1):
                result = "Advantage " + self.player2_name
            elif (minus_result >= 2):
                result = "Win for " + self.player1_name
            else:
                result = "Win for " + self.player2_name
        else:
            for i in range(1, 3):
                if (i == 1):
                    temp_score = self.p1_points
                else:
                    result += "-"
                    temp_score = self.p2_points
                result += {
                    0: "Love",
                    1: "Fifteen",
                    2: "Thirty",
                    3: "Forty",
                }[temp_score]
        return result


class TennisGameDefactored2:
    def __init__(self, player1_name, player2_name):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.p1_points = 0
        self.p2_points = 0

    def won_point(self, player_name):
        if player_name == self.player1_name:
            self.p1_score()
        else:
            self.p2_score()

    def score(self):
        result = ""
        if (self.p1_points == self.p2_points and self.p1_points < 4):
            if (self.p1_points == 0):
                result = "Love"
            if (self.p1_points == 1):
                result = "Fifteen"
            if (self.p1_points == 2):
                result = "Thirty"
            if (self.p1_points == 3):
                result = "Forty"
            result += "-All"
        if (self.p1_points == self.p2_points and self.p1_points > 3):
            result = "Deuce"

        p1_res = ""
        p2_res = ""
        if (self.p1_points > 0 and self.p2_points == 0):
            if (self.p1_points == 1):
                p1_res = "Fifteen"
            if (self.p1_points == 2):
                p1_res = "Thirty"
            if (self.p1_points == 3):
                p1_res = "Forty"

            p2_res = "Love"
            result = p1_res + "-" + p2_res
        if (self.p2_points > 0 and self.p1_points == 0):
            if (self.p2_points == 1):
                p2_res = "Fifteen"
            if (self.p2_points == 2):
                p2_res = "Thirty"
            if (self.p2_points == 3):
                p2_res = "Forty"

            p1_res = "Love"
            result = p1_res + "-" + p2_res

        if (self.p1_points > self.p2_points and self.p1_points < 4):
            if (self.p1_points == 2):
                p1_res = "Thirty"
            if (self.p1_points == 3):
                p1_res = "Forty"
            if (self.p2_points == 1):
                p2_res = "Fifteen"
            if (self.p2_points == 2):
                p2_res = "Thirty"
            result = p1_res + "-" + p2_res
        if (self.p2_points > self.p1_points and self.p2_points < 4):
            if (self.p2_points == 2):
                p2_res = "Thirty"
            if (self.p2_points == 3):
                p2_res = "Forty"
            if (self.p1_points == 1):
                p1_res = "Fifteen"
            if (self.p1_points == 2):
                p1_res = "Thirty"
            result = p1_res + "-" + p2_res

        if (self.p1_points > self.p2_points and self.p2_points >= 3):
            result = "Advantage " + self.player1_name

        if (self.p2_points > self.p1_points and self.p1_points >= 3):
            result = "Advantage " + self.player2_name

        if (self.p1_points >= 4 and self.p2_points >= 0 and (self.p1_points-self.p2_points) >= 2):
            result = "Win for " + self.player1_name
        if (self.p2_points >= 4 and self.p1_points >= 0 and (self.p2_points-self.p1_points) >= 2):
            result = "Win for " + self.player2_name
        return result

    def set_p1_score(self, number):
        for i in range(number):
            self.p1_score()

    def set_p2_score(self, number):
        for i in range(number):
            self.p2_score()

    def p1_score(self):
        self.p1_points += 1

    def p2_score(self):
        self.p2_points += 1


class TennisGameDefactored3:
    def __init__(self, player1_name, player2_name):
        self.p1_n = player1_name
        self.p2_n = player2_name
        self.p1 = 0
        self.p2 = 0

    def won_point(self, n):
        if n == self.p1_n:
            self.p1 += 1
        else:
            self.p2 += 1

    def score(self):
        if (self.p1 < 4 and self.p2 < 4):
            p = ["Love", "Fifteen", "Thirty", "Forty"]
            s = p[self.p1]
            return s + "-All" if (self.p1 == self.p2) else s + "-" + p[self.p2]
        else:
            if (self.p1 == self.p2):
                return "Deuce"
            s = self.p1_n if self.p1 > self.p2 else self.p2_n
            return "Advantage " + s if ((self.p1-self.p2)*(self.p1-self.p2) == 1) else "Win for " + s


# NOTE: You must change this to point at the one of the three examples that you're working on!
TennisGame = TennisGameDefactored1
