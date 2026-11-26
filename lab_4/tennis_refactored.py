# -*- coding: utf-8 -*-
GAME_SCORE = {
    0 : "Love",
    1 : "Fifteen",
    2 : "Thirty",
    3 : "Forty",
}

POINTS_TO_WIN = 4


class Player:

    def __init__(self, name):
        self.name = name
        self.points = 0

    def add_point(self):
        self.points += 1


class TennisGameDefactored1:

    def __init__(self, player_1_name, player_2_name):
        self.player_1 = Player(player_1_name)
        self.player_2 = Player(player_2_name)
        
    def won_point(self, player_name):
        if player_name == self.player_1.name:
            self.player_1.add_point()
        else:
            self.player_2.add_point()
    
    def score(self):
        if self.player_1.points == self.player_2.points:
            equal_points = self.player_1.points

            if equal_points >= POINTS_TO_WIN:
                return "Deuce"
            return f"{GAME_SCORE[equal_points]}-All"

        if self.player_1.points >= POINTS_TO_WIN or self.player_2.points >= POINTS_TO_WIN:
            points_diff = self.player_1.points - self.player_2.points
            leader = self.player_1 if points_diff > 0 else self.player_2

            if abs(points_diff) == 1:
                return f"Advantage {leader.name}"
            return f"Win for {leader.name}"

        return f"{GAME_SCORE[self.player_1.points]}-{GAME_SCORE[self.player_2.points]}"


class TennisGameDefactored2:
    def __init__(self, player_1_name, player_2_name):
        self.player_1 = Player(player_1_name)
        self.player_2 = Player(player_2_name)

    def won_point(self, player_name):
        if player_name == self.player_1.name:
            self.player_1.add_point()
        else:
            self.player_2.add_point()
    
    def score(self):
        score_diff = self.player_1.points - self.player_2.points

        if not score_diff:
            if self.player_1.points >= POINTS_TO_WIN:
                return "Deuce"
            return f"{GAME_SCORE[self.player_1.points]}-All"

        if self.player_1.points < POINTS_TO_WIN and self.player_2.points < POINTS_TO_WIN:
            return f"{GAME_SCORE[self.player_1.points]}-{GAME_SCORE[self.player_2.points]}"

        leader = self.player_1 if score_diff > 0 else self.player_2

        if abs(score_diff) == 1:
            return f"Advantage {leader.name}"
        return f"Win for {leader.name}"
        

class TennisGameDefactored3:
    def __init__(self, player1Name, player2Name):
        self.p1N = player1Name
        self.p2N = player2Name
        self.p1 = 0
        self.p2 = 0
        
    def won_point(self, n):
        if n == self.p1N:
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
            s = self.p1N if self.p1 > self.p2 else self.p2N
            return "Advantage " + s if ((self.p1-self.p2)*(self.p1-self.p2) == 1) else "Win for " + s

# NOTE: You must change this to point at the one of the three examples that you're working on!
TennisGame = TennisGameDefactored2