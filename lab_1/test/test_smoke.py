from file_iterator import FileIterator


def test_file_iterator(tmp_path) -> None:
    f = tmp_path / "sample.txt"

    f.write_text("line1\nline2\n", encoding="utf-8")

    lines = list(FileIterator(str(f)))

    assert lines == ["line1\n", "line2\n"]
