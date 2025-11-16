import json
from card_finder import serialization


def test_serialization(tmp_path):
    file = tmp_path / "result.json"
    serialization("123456", file)

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["card_numbers"] == "123456"
