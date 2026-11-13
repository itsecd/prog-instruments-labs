"""Tennis scoring logic."""


class TennisGame:
    """Calculate the score of a tennis game for two players."""

    _POINT_NAMES = ("Love", "Fifteen", "Thirty", "Forty")

    def __init__(self, player1: str, player2: str) -> None:
        self.player1 = player1
        self.player2 = player2
        self._points = {player1: 0, player2: 0}

    def won_point(self, player: str) -> None:
        """Add one point to the named player."""
        if player not in self._points:
            raise ValueError(f"Unknown player: {player}")
        self._points[player] += 1

    def score(self) -> str:
        """Return the current tennis score."""
        first = self._points[self.player1]
        second = self._points[self.player2]

        if first == second:
            if first >= 3:
                return "Deuce"
            return f"{self._POINT_NAMES[first]}-All"

        if first >= 4 or second >= 4:
            difference = first - second
            if difference == 1:
                return f"Advantage {self.player1}"
            if difference == -1:
                return f"Advantage {self.player2}"
            if difference > 1:
                return f"Win for {self.player1}"
            return f"Win for {self.player2}"

        return f"{self._POINT_NAMES[first]}-{self._POINT_NAMES[second]}"
