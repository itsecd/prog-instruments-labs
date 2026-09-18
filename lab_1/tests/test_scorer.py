import scorer
from scorer import get_score


def test_scorer():
    scorer.flavour = None
    assert get_score() == -1
