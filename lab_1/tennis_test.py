import pytest

from tennis_unittest import play_game, test_cases


class TestTennis:
    @pytest.mark.parametrize(
        ["p1Points", "p2Points", "score", "p1Name", "p2Name"], test_cases
    )
    def test_get_score(self, p1Points, p2Points, score, p1Name, p2Name):
        game = play_game(p1Points, p2Points, p1Name, p2Name)
        assert score == game.score()
