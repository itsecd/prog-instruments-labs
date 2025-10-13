# -*- coding: utf-8 -*-

class TennisGameDefactored1:

    SCORE_TO_TEXT = {
    0: "Love",
    1: "Fifteen",
    2: "Thirty",
    3: "Forty"
}

    def _get_score_text(self, score):
        return self.SCORE_TO_TEXT.get(score, "Deuce")

    def __init__(self, player1Name, player2Name):
        self.player1Name = player1Name
        self.player2Name = player2Name
        self.p1points = 0
        self.p2points = 0
        
    def won_point(self, playerName):
        if playerName == self.player1Name:
            self.p1points += 1
        else:
            self.p2points += 1
    
    def _format_equal_score(self):
        if self.p1points >= 3:
            return "Deuce"
        return f"{self._get_score_text(self.p1points)}-All"

    def _format_advantage_or_win(self):
        score_diff = self.p1points - self.p2points
        if abs(score_diff) >= 2:
            winner = self.player1Name if score_diff > 0 else self.p2points
            return f"Win for {winner}"
        leader = self.player1Name if score_diff > 0 else self.p2points
        return f"Advantage {leader}"

    def _format_regular_score(self):
        p1_text = self._get_score_text(self.p1points)
        p2_text = self._get_score_text(self.p2points)
        return f"{p1_text}-{p2_text}"

    def score(self):
        if self.p1points == self.p2points:
            return self._format_equal_score()
        if self.p1points >= 4 or self.p2points >= 4:
            return self._format_advantage_or_win()
        return self._format_regular_score()


class TennisGameDefactored2:
    SCORE_TO_TEXT = {
        0: "Love",
        1: "Fifteen",
        2: "Thirty",
        3: "Forty"
    }

    def __init__(self, player1Name, player2Name):
        self.player1Name = player1Name
        self.player2Name = player2Name
        self.p1points = 0
        self.p2points = 0
        
    def award_point(self, player_name):
        if player_name == self.player1Name:
            self.p1points += 1
        else:
            self.p2points += 1
    
    def _get_score_text(self, score):
        return self.SCORE_TO_TEXT.get(score, "Deuce")

    def _format_equal_score(self):
        if self.p1points >= 3:
            return "Deuce"
        return f"{self._get_score_text(self.p1points)}-All"

    def _format_advantage_or_win(self):
        score_diff = self.p1points - self.p2points
        if abs(score_diff) >= 2:
            winner = self.player1Name if score_diff > 0 else self.player2Name
            return f"Win for {winner}"
        leader = self.player1Name if score_diff > 0 else self.player2Name
        return f"Advantage {leader}"

    def _format_regular_score(self):
        p1_text = self._get_score_text(self.p1points)
        p2_text = self._get_score_text(self.p2points)
        return f"{p1_text}-{p2_text}"

    def score(self):
        if self.p1points == self.p2points:
            return self._format_equal_score()
        if self.p1points >= 4 or self.p2points >= 4:
            return self._format_advantage_or_win()
        return self._format_regular_score()
    
    def P1Score(self):
        self.p1points +=1
    
    
    def P2Score(self):
        self.p2points +=1
        
class TennisGameDefactored3:

    SCORE_TO_TEXT = {
        0: "Love",
        1: "Fifteen",
        2: "Thirty",
        3: "Forty"
    }

    def __init__(self, player1Name, player2Name):
        self.player1Name = player1Name
        self.player2Name = player2Name
        self.p1points = 0
        self.p2points = 0
        
    def award_point(self, player_name):
        if player_name == self.player1Name:
            self.p1points += 1
        else:
            self.p2points += 1
    
    def _get_score_text(self, score):
        return self.SCORE_TO_TEXT.get(score, "Deuce")
    
    def _format_equal_score(self):
        if self.p1points >= 3:
            return "Deuce"
        return f"{self._get_score_text(self.p1points)}-All"

    def _format_advantage_or_win(self):
        score_diff = self.p1points - self.p2points
        if abs(score_diff) >= 2:
            winner = self.player1Name if score_diff > 0 else self.player2Name
            return f"Win for {winner}"
        leader = self.player1Name if score_diff > 0 else self.player2Name
        return f"Advantage {leader}"

    def _format_regular_score(self):
        p1_text = self._get_score_text(self.p1points)
        p2_text = self._get_score_text(self.p2points)
        return f"{p1_text}-{p2_text}"

    def score(self):
        if self.p1points == self.p2points:
            return self._format_equal_score()
        if self.p1points >= 4 or self.p2points >= 4:
            return self._format_advantage_or_win()
        return self._format_regular_score()

# NOTE: You must change this to point at the one of the three examples that you're working on!
TennisGame = TennisGameDefactored1
