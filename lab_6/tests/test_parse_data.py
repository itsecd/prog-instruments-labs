
from main import parse_data


def test_parse_data_sorts_by_last_name():
    data = "Фамилия: Петров\nИмя: Иван\n2)\nФамилия: Иванов\nИмя: Алексей"

    result = parse_data(data)

    assert result == [
        "Фамилия: Иванов\nИмя: Алексей",
        "Фамилия: Петров\nИмя: Иван",
    ]


def test_parse_data_sorts_by_first_name():
    data = "Фамилия: Петров\nИмя: Борис\n2)\nФамилия: Петров\nИмя: Алексей"

    result = parse_data(data)

    assert result == [
        "Фамилия: Петров\nИмя: Алексей",
        "Фамилия: Петров\nИмя: Борис",
    ]


def test_parse_data_skips_incomplete_forms():
    data = "Фамилия: Петров\nВозраст: 20\n2)\nФамилия: Иванов\nИмя: Олег"

    result = parse_data(data)

    assert result == ["Фамилия: Иванов\nИмя: Олег"]
