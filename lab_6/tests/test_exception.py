import pytest

from src.exception import CalcException


def test_calc_exception_is_an_exception():
    assert issubclass(CalcException, Exception)


def test_message_is_added_to_string_representation():
    error = CalcException("Trying to divide by zero")
    assert error.message == "Trying to divide by zero"
    assert str(error) == "CalcException: Trying to divide by zero"


def test_default_message_is_used_when_no_arguments_given():
    assert str(CalcException()) == "CalcException: Something went wrong ..."


def test_calc_exception_can_be_raised_and_caught():
    with pytest.raises(CalcException, match="boom"):
        raise CalcException("boom")