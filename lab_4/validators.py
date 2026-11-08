"""Pure validators that go beyond a single regex."""
from datetime import datetime

from patterns import PATTERNS


def is_valid_date(value: str) -> bool:
    """Return True iff value is a real calendar date in YYYY-MM-DD form."""
    if PATTERNS["date"].fullmatch(value) is None:
        return False
    try:
        # A date has no time component; timezone is not applicable here.
        datetime.strptime(value, "%Y-%m-%d")  # noqa: DTZ007
    except ValueError:
        return False
    return True