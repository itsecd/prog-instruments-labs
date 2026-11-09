"""Registry of the ten variant-26 CSV columns and how to validate them."""

from collections.abc import Callable
from dataclasses import dataclass

from patterns import PATTERNS
from validators import is_valid_date


@dataclass(frozen=True)
class Column:
    name: str
    pattern: str  # stored for documentation/debugging
    validator: Callable[[str], bool]


def _regex_validator(name: str) -> Callable[[str], bool]:
    compiled = PATTERNS[name]
    return lambda value: compiled.fullmatch(value) is not None


COLUMNS: tuple[Column, ...] = (
    Column("telephone", PATTERNS["telephone"].pattern, _regex_validator("telephone")),
    Column(
        "http_status_message",
        PATTERNS["http_status_message"].pattern,
        _regex_validator("http_status_message"),
    ),
    Column("snils", PATTERNS["snils"].pattern, _regex_validator("snils")),
    Column(
        "identifier", PATTERNS["identifier"].pattern, _regex_validator("identifier")
    ),
    Column("ip_v4", PATTERNS["ip_v4"].pattern, _regex_validator("ip_v4")),
    Column("longitude", PATTERNS["longitude"].pattern, _regex_validator("longitude")),
    Column(
        "blood_type", PATTERNS["blood_type"].pattern, _regex_validator("blood_type")
    ),
    Column("isbn", PATTERNS["isbn"].pattern, _regex_validator("isbn")),
    Column(
        "locale_code", PATTERNS["locale_code"].pattern, _regex_validator("locale_code")
    ),
    Column("date", PATTERNS["date"].pattern, is_valid_date),
)

COLUMN_NAMES: tuple[str, ...] = tuple(c.name for c in COLUMNS)
