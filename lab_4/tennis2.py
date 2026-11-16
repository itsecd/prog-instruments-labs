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
        if player_name == self.player1_name:
            self.p1_score()
        else:
            self.p2_score()

    def _get_score_name(self, points):
        if points < 4:
            return self.SCORE_NAMES[points]
        return ""

    def score(self):
        if self._is_win(self.p1points, self.p2points):
            return self.WIN_P1
        if self._is_win(self.p2points, self.p1points):
            return self.WIN_P2
            
        if self._is_deuce():
            return self.DEUCE
            
        if self._is_advantage(self.p1points, self.p2points):
            return self.ADVANTAGE_P1
        if self._is_advantage(self.p2points, self.p1points):
            return self.ADVANTAGE_P2
            
        if self.p1points == self.p2points:
            return self._get_score_name(self.p1points) + "-All"
            
        return self._get_score_name(self.p1points) + "-" + self._get_score_name(self.p2points)

    def _is_win(self, p1_pts, p2_pts):
        return p1_pts >= 4 and (p1_pts - p2_pts) >= 2

    def _is_deuce(self):
        return self.p1points == self.p2points and self.p1points >= 3

    def _is_advantage(self, p1_pts, p2_pts):
        return p1_pts > p2_pts and p2_pts >= 3


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