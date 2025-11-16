from card_finder import find_card_number


def test_find_card_number(monkeypatch):
    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def imap_unordered(self, func, data, chunksize=500):
            yield None
            yield "FOUND_CARD"

        def close(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr("multiprocessing.Pool", lambda size: FakePool())

    monkeypatch.setattr("card_finder.gen_card_nums", lambda bin_code: ["1111", "2222"])

    result = find_card_number()
    assert result == "FOUND_CARD"
