import pytest
from main import _is_valid_date, validate_csv

@pytest.mark.parametrize("date_str, is_valid", [
    ("2026-11-08", True),
    ("2024-02-29", True),
    ("2026-02-29", False),
    ("2026-13-01", False),
    ("2026-11-08_extra", False),
    ("", False),
])
def test_is_valid_date(date_str, is_valid):
    assert _is_valid_date(date_str) is is_valid

def test_validate_csv_with_tmp_path(tmp_path):
    csv_content = (
        "telephone;http_status_message;snils;identifier;ip_v4;longitude;blood_type;isbn;locale_code;date\n"

        "+7-(999)-123-45-67;200 OK;12345678901;12-34/56;192.168.1.1;45.0;A+;1-23456-789-0;en-us;2026-11-08\n"

        "+7-(999)-123-45-68;404 Not Found;12345678902;12-34/57;192.168.1.2;45.1;B−;1-23456-789-1;ru-ru;2024-02-29\n"

        "INVALID_PHONE;500 ERROR;12345678903;12-34/58;192.168.1.3;45.2;O+;2-34567-890-1;en-gb;2026-11-10\n"
    )
    
    test_csv = tmp_path / "test_data.csv"
    test_csv.write_text(csv_content, encoding="utf-16")

    result = validate_csv(test_csv, encoding="utf-16")

    assert result.row_count == 3
    assert len(result.invalid_rows) == 1
    assert result.invalid_rows[0] == 2
    assert result.errors_by_field["telephone"] == 1