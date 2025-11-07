import pytest

from parser import arg_parse


def test_parser_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test successful argument parsing
    :param monkeypatch: pytest fixture for modifying objects
    """
    monkeypatch.setattr('sys.argv', ['script', 'test.csv'])
    result: str = arg_parse()
    assert result == 'test.csv'


def test_parser_missing_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test missing argument handling
    :param monkeypatch: pytest fixture for modifying objects
    """ 
    monkeypatch.setattr('sys.argv', ['script'])
    with pytest.raises(SystemExit):
        arg_parse()