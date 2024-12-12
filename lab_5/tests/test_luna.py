import hashlib
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from luna import Luna


def test_Luna_valid_card():
    """
    ????????? ??????? Luna ? ?????????????? ??????? ?????.
    """
    valid_card = "4274020236520877"
    assert Luna(valid_card)  

def test_Luna_invalid_card():
    """
    ????????? ??????? Luna ? ???????????????? ??????? ?????.
    """
    invalid_card = "1234567890123456"
    assert not Luna(invalid_card)  