from yahtzee import Yahtzee


def test_chance_scores_sum_of_all_dice() -> None:
    """Tests that the chance method calculates the sum of all dice correctly."""
    expected: int = 15
    actual: int = Yahtzee.chance(2, 3, 4, 5, 1)
    assert expected == actual
    assert 16 == Yahtzee.chance(3, 3, 4, 5, 1)


def test_yahtzee_scores_50() -> None:
    """Tests that the Yahtzee method scores 50 when all dice are the same, and 0 otherwise."""
    expected: int = 50
    actual: int = Yahtzee.yahtzee([4, 4, 4, 4, 4])
    assert expected == actual
    assert 50 == Yahtzee.yahtzee([6, 6, 6, 6, 6])
    assert 0 == Yahtzee.yahtzee([6, 6, 6, 6, 3])


def test_ones() -> None:
    """Tests that the ones method returns the correct sum of all dice showing the number 1."""
    assert Yahtzee.ones(1, 2, 3, 4, 5) == 1
    assert 2 == Yahtzee.ones(1, 2, 1, 4, 5)
    assert 0 == Yahtzee.ones(6, 2, 2, 4, 5)
    assert 4 == Yahtzee.ones(1, 2, 1, 1, 1)


def test_twos() -> None:
    """Tests that the twos method returns the correct sum of all dice showing the number 2."""
    assert 4 == Yahtzee.twos(1, 2, 3, 2, 6)
    assert 10 == Yahtzee.twos(2, 2, 2, 2, 2)


def test_threes() -> None:
    """Tests that the threes method returns the correct sum of all dice showing the number 3."""
    assert 6 == Yahtzee.threes(1, 2, 3, 2, 3)
    assert 12 == Yahtzee.threes(2, 3, 3, 3, 3)


def test_fours() -> None:
    """Tests that the fours method returns the correct sum of all dice showing the number 4."""
    assert 12 == Yahtzee(4, 4, 4, 5, 5).fours()
    assert 8 == Yahtzee(4, 4, 5, 5, 5).fours()
    assert 4 == Yahtzee(4, 5, 5, 5, 5).fours()


def test_fives() -> None:
    """Tests that the fives method returns the correct sum of all dice showing the number 5."""
    assert 10 == Yahtzee(4, 4, 4, 5, 5).fives()
    assert 15 == Yahtzee(4, 4, 5, 5, 5).fives()
    assert 20 == Yahtzee(4, 5, 5, 5, 5).fives()


def test_sixes() -> None:
    """Tests that the sixes method returns the correct sum of all dice showing the number 6."""
    assert 0 == Yahtzee(4, 4, 4, 5, 5).sixes()
    assert 6 == Yahtzee(4, 4, 6, 5, 5).sixes()
    assert 18 == Yahtzee(6, 5, 6, 6, 5).sixes()


def test_one_pair() -> None:
    """Tests that the score_pair method returns the correct score for a single pair of dice."""
    assert 6 == Yahtzee.score_pair(3, 4, 3, 5, 6)
    assert 10 == Yahtzee.score_pair(5, 3, 3, 3, 5)
    assert 12 == Yahtzee.score_pair(5, 3, 6, 6, 5)


def test_two_pair() -> None:
    """Tests that the two_pair method returns the correct score for two pairs of dice."""
    assert 16 == Yahtzee.two_pair(3, 3, 5, 4, 5)
    assert 0 == Yahtzee.two_pair(3, 3, 5, 5, 5)


def test_three_of_a_kind() -> None:
    """Tests that the three_of_a_kind method returns the correct score for three of a kind."""
    assert 9 == Yahtzee.three_of_a_kind(3, 3, 3, 4, 5)
    assert 15 == Yahtzee.three_of_a_kind(5, 3, 5, 4, 5)
    assert 0 == Yahtzee.three_of_a_kind(3, 3, 3, 3, 5)


def test_four_of_a_kind() -> None:
    """Tests that the four_of_a_kind method returns the correct score for four of a kind."""
    assert 12 == Yahtzee.four_of_a_kind(3, 3, 3, 3, 5)
    assert 20 == Yahtzee.four_of_a_kind(5, 5, 5, 4, 5)
    assert 0 == Yahtzee.three_of_a_kind(3, 3, 3, 3, 3)


def test_small_straight() -> None:
    """Tests that the small_straight method scores 15 for a small straight."""
    assert 15 == Yahtzee.small_straight(1, 2, 3, 4, 5)
    assert 15 == Yahtzee.small_straight(2, 3, 4, 5, 1)
    assert 0 == Yahtzee.small_straight(1, 2, 2, 4, 5)


def test_large_straight() -> None:
    """Tests that the large_straight method scores 20 for a large straight."""
    assert 20 == Yahtzee.large_straight(6, 2, 3, 4, 5)
    assert 20 == Yahtzee.large_straight(2, 3, 4, 5, 6)
    assert 0 == Yahtzee.large_straight(1, 2, 2, 4, 5)


def test_full_house() -> None:
    """Tests that the full_house method scores the sum of a full house."""
    assert 18 == Yahtzee.full_house(6, 2, 2, 2, 6)
    assert 0 == Yahtzee.full_house(2, 3, 4, 5, 6)
