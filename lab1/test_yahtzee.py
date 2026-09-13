from yahtzee import Yahtzee


def test_chance_scores_sum_of_all_dice() -> None:
    expected = 15
    actual = Yahtzee.chance(2, 3, 4, 5, 1)
    assert expected == actual
    assert Yahtzee.chance(3, 3, 4, 5, 1) == 16


def test_yahtzee_scores_50() -> None:
    expected = 50
    actual = Yahtzee.yahtzee([4, 4, 4, 4, 4])
    assert expected == actual
    assert Yahtzee.yahtzee([6, 6, 6, 6, 6]) == 50
    assert Yahtzee.yahtzee([6, 6, 6, 6, 3]) == 0


def test_1s() -> None:
    assert Yahtzee.ones(1, 2, 3, 4, 5) == 1
    assert Yahtzee.ones(1, 2, 1, 4, 5) == 2
    assert Yahtzee.ones(6, 2, 2, 4, 5) == 0
    assert Yahtzee.ones(1, 2, 1, 1, 1) == 4


def test_2s() -> None:
    assert Yahtzee.twos(1, 2, 3, 2, 6) == 4
    assert Yahtzee.twos(2, 2, 2, 2, 2) == 10


def test_threes() -> None:
    assert Yahtzee.threes(1, 2, 3, 2, 3) == 6
    assert Yahtzee.threes(2, 3, 3, 3, 3) == 12


def test_fours() -> None:
    assert Yahtzee(4, 4, 4, 5, 5).fours() == 12
    assert Yahtzee(4, 4, 5, 5, 5).fours() == 8
    assert Yahtzee(4, 5, 5, 5, 5).fours() == 4


def test_fives() -> None:
    assert Yahtzee(4, 4, 4, 5, 5).fives() == 10
    assert Yahtzee(4, 4, 5, 5, 5).fives() == 15
    assert Yahtzee(4, 5, 5, 5, 5).fives() == 20


def test_sixes() -> None:
    assert Yahtzee(4, 4, 4, 5, 5).sixes() == 0
    assert Yahtzee(4, 4, 6, 5, 5).sixes() == 6
    assert Yahtzee(6, 5, 6, 6, 5).sixes() == 18


def test_one_pair() -> None:
    assert Yahtzee.score_pair(3, 4, 3, 5, 6) == 6
    assert Yahtzee.score_pair(5, 3, 3, 3, 5) == 10
    assert Yahtzee.score_pair(5, 3, 6, 6, 5) == 12


def test_two_pair() -> None:
    assert Yahtzee.two_pair(3, 3, 5, 4, 5) == 16
    assert Yahtzee.two_pair(3, 3, 5, 5, 5) == 0


def test_three_of_a_kind() -> None:
    assert Yahtzee.three_of_a_kind(3, 3, 3, 4, 5) == 9
    assert Yahtzee.three_of_a_kind(5, 3, 5, 4, 5) == 15
    assert Yahtzee.three_of_a_kind(3, 3, 3, 3, 5) == 0


def test_four_of_a_kind() -> None:
    assert Yahtzee.four_of_a_kind(3, 3, 3, 3, 5) == 12
    assert Yahtzee.four_of_a_kind(5, 5, 5, 4, 5) == 20
    assert Yahtzee.three_of_a_kind(3, 3, 3, 3, 3) == 0


def test_small_straight() -> None:
    assert Yahtzee.small_straight(1, 2, 3, 4, 5) == 15
    assert Yahtzee.small_straight(2, 3, 4, 5, 1) == 15
    assert Yahtzee.small_straight(1, 2, 2, 4, 5) == 0


def test_large_straight() -> None:
    assert Yahtzee.large_straight(6, 2, 3, 4, 5) == 20
    assert Yahtzee.large_straight(2, 3, 4, 5, 6) == 20
    assert Yahtzee.large_straight(1, 2, 2, 4, 5) == 0


def test_full_house() -> None:
    assert Yahtzee.full_house(6, 2, 2, 2, 6) == 18
    assert Yahtzee.full_house(2, 3, 4, 5, 6) == 0
