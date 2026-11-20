import pytest

import lab1_var18 as app


@pytest.mark.parametrize(
    ("phone", "expected"),
    [
        ("+7 (495) 123-45-67", True),
        ("8 495 123 45 67", True),
        ("84951234567", True),
        ("+7 495 123 45", False),
        ("+1 202 555 0142", False),
        ("+7 495 ABC-45-67", False),
        ("", False),
    ],
)
def test_is_valid_phone_accepts_only_supported_formats(phone, expected):
    assert app.is_valid_phone(phone) is expected

def _profile(number, contact):
    return [
        f"{number})",
        "Имя: Тест",
        "Возраст: 25",
        "Город: Самара",
        f"Номер телефона или email: {contact}",
        "Профессия: Инженер",
        "Комментарий: Пример",
    ]


def test_parse_profiles_skips_noise_and_incomplete_profile():
    first = _profile(1, "+7 (495) 123-45-67")
    second = _profile(2, "123")
    text = (
        "Список анкет\n\n"
        + "\n".join(f"  {line}  " for line in first)
        + "\n\n"
        + "\n".join(second)
        + "\n\n3)\nИмя: Незавершённая анкета"
    )

    assert app.parse_profiles(text) == [first, second]


def test_find_invalid_phones_selects_only_invalid_phone():
    valid = _profile(1, "+7 (495) 123-45-67")
    email = _profile(2, "test@example.org")
    invalid = _profile(3, "123")

    assert app.find_invalid_phones([valid, email, invalid]) == [invalid]


def test_read_file_reports_missing_file(tmp_path):
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="не найден") as error:
        app.read_file(missing)

    assert str(missing) in str(error.value)


def test_read_file_preserves_cause_of_io_error(tmp_path, monkeypatch):
    input_path = tmp_path / "profiles.txt"
    permission_error = PermissionError("Permission denied")

    def fail_to_read(path, *, encoding):
        raise permission_error

    with monkeypatch.context() as patch:
        patch.setattr(type(input_path), "read_text", fail_to_read)

        with pytest.raises(OSError, match="Ошибка при чтении файла") as error:
            app.read_file(input_path)

    assert str(input_path) in str(error.value)
    assert error.value.__cause__ is permission_error


def test_process_profiles_writes_only_valid_profiles(tmp_path, capsys):
    valid = _profile(1, "84951234567")
    invalid = _profile(2, "123")
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    input_path.write_text(
        "\n".join(valid) + "\n\n" + "\n".join(invalid),
        encoding="utf-8",
    )

    counts = app.process_profiles(input_path, output_path)

    assert counts == (2, 1, 1)
    assert output_path.read_text(encoding="utf-8") == "\n".join(valid)
    assert "Некорректный номер телефона: 123" in capsys.readouterr().out


def test_process_profiles_keeps_layout_when_all_profiles_are_valid(tmp_path):
    original = "Список анкет\n\n" + "\n".join(
        _profile(1, "test@example.org")
    ) + "\n\n"
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    input_path.write_text(original, encoding="utf-8")

    assert app.process_profiles(input_path, output_path) == (1, 0, 1)
    assert output_path.read_bytes() == input_path.read_bytes()