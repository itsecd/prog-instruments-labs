import pytest
from types import SimpleNamespace
from src.utils import limit_string, dict_to_sn


def test_limit_string():
    string = "Very long and not short message"
    limit = 20
    expected = "Very long and not..."

    actual = limit_string(string, limit)

    assert expected == actual


@pytest.mark.parametrize(
    "string, limit",
    [
        ("goodstring", -1),
        ("goodstring", 2),
        ("goodstring", 3),
        ("", 10),
        ("b", 10),
        ("ba", 10),
        ("bad", 10),
    ],
)
def test_limit_string_exception(string: str, limit: int):
    with pytest.raises(ValueError):
        limit_string(string, limit)


def test_dict_to_sn():
    dict_ = {
        "messages": {
            "cringe": "std::idno",
            "bad": "PLUWo",
        },
    }
    expected = SimpleNamespace(
        messages=SimpleNamespace(
            cringe="std::idno",
            bad="PLUWo",
        ),
    )

    actual = dict_to_sn(dict_)

    assert actual == expected
