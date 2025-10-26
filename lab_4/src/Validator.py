import re
from Entry import Entry


class Validator:
    """Validator class to create object with get_valid / get_invalid methods"""
    entries: list[Entry]

    def __init__(self, entries: list[Entry]):
        """Constructor: gets entries and copy them to local list"""
        self.entries = []

        for i in entries:
            self.entries.append(i.copy())

    def parse_invalid(self) -> dict:
        """Get invalid writes"""
        illegal_entries = {
            "telephone": 0,
            "height": 0,
            "inn": 0,
            "passport_number": 0,
            "university": 0,
            "age": 0,
            "academic_degree": 0,
            "worldview": 0,
            "address": 0
        }

        for i in self.entries:
            illkeys = self.parse_entry(i)

            for j in illkeys:
                illegal_entries[j] += 1

        return illegal_entries

    def parse_valid(self) -> list[Entry]:
        """Get valid writes"""
        legal_entries: list[Entry] = []

        for i in self.entries:
            illkeys = self.parse_entry(i)

            if len(illkeys) == 0:
                legal_entries.append(i)

        return legal_entries

    def parse_entry(self, entry: Entry) -> list[str]:
        """Universal entry validation"""
        illegal_keys = []

        rules = {
            "telephone": (
            r"\+[0-9]-\([0-9]{3}\)-[0-9]{3}-[0-9]{2}-[0-9]{2}", "regex"),
            "inn": (r"^\d{12}$", "regex"),
            "passport_number": (6, "len"),
            "height": ((1.2, 2.2), "range_float"),
            "age": ((18, 110), "range_int"),
            "address": (r".+\d+", "regex"),
            "university": (r"[а-яА-Я]+", "regex"),
            "academic_degree": (r"[a-zA-Zа-яА-Я]+", "regex"),
            "worldview": (r"[a-zA-Zа-яА-Я]+", "regex")
        }

        for field, (rule, rule_type) in rules.items():
            value = getattr(entry, field)
            if not self.validate(value, rule, rule_type):
                illegal_keys.append(field)

        return illegal_keys

    def validate(self, value: str, rule, rule_type: str) -> bool:
        """Generic field validation"""
        try:
            if rule_type == "regex":
                return bool(re.match(rule, value))
            elif rule_type == "len":
                return len(str(value)) == rule
            elif rule_type == "range_float":
                low, high = rule
                return low < float(value) < high
            elif rule_type == "range_int":
                low, high = rule
                v = int(value)
                return low <= v < high
            else:
                return True
        except Exception:
            return False
