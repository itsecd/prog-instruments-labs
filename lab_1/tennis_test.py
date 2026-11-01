import pytest

from tennis_unittest import play_game, test_cases


class TestTennis:
    @pytest.mark.parametrize(
        ("p1_points", "p2_points", "score", "p1_name", "p2_name"), test_cases
    )
    def test_get_score(
        self, p1_points: int, p2_points: int, score: str, p1_name: str, p2_name: str
    ) -> None:
        game = play_game(p1_points, p2_points, p1_name, p2_name)
        assert score == game.score()
