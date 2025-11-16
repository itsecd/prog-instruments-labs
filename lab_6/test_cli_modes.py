from main import mode_brute_force, mode_luhn_check, mode_benchmark


def test_mode_brute_force(monkeypatch):
    calls = {"saved": None}

    monkeypatch.setattr("main.find_card_number", lambda: "CARD123")
    monkeypatch.setattr("main.serialization", lambda card: calls.update(saved=card))

    mode_brute_force()
    assert calls["saved"] == "CARD123"


def test_mode_luhn_check(monkeypatch):
    calls = {"checked": None}

    monkeypatch.setattr("main.find_card_number", lambda: "4111111111111111")
    monkeypatch.setattr("main.is_valid", lambda card: calls.update(checked=card))

    mode_luhn_check()
    assert calls["checked"] == "4111111111111111"


def test_mode_benchmark(monkeypatch):
    calls = {"drawn": False}

    monkeypatch.setattr("main.collect_times", lambda: ([1, 2], [0.5, 0.3]))
    monkeypatch.setattr("main.get_timing_graph", lambda a, b: calls.update(drawn=True))

    mode_benchmark()
    assert calls["drawn"] is True
