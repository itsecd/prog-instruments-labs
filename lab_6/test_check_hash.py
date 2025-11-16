import pytest
import hashlib
from card_finder import check_hash
from consts import CARD_HASH


@pytest.mark.parametrize("card_num", [
    "0000000000000000",
    "1111111111111111",
    "9999999999999999",
])
def test_hash_not_matching(card_num):
    assert check_hash(card_num) is None


def test_hash_matching(monkeypatch):
    # fake sha3_256, always returns CARD_HASH
    class FakeHash:
        def hexdigest(self):
            return CARD_HASH

    def fake_sha3(data):
        return FakeHash()

    monkeypatch.setattr(hashlib, "sha3_256", fake_sha3)

    result = check_hash("FAKECARD")
    assert result == "FAKECARD"
