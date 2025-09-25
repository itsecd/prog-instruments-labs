from src.utils import limit_string


def test_limit_string():
    string = "Very long and not short message"
    limit = 20
    expected = "Very long and not..."

    actual = limit_string(string, limit)

    assert limit == len(actual)
    assert expected == actual
