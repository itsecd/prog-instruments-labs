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
        """Parse simple node"""
        illegal_keys = []

        if not self.check_telephone(entry['telephone']):
            illegal_keys.append('telephone')
        elif not self.check_inn(entry['inn']):
            illegal_keys.append('inn')
        elif not self.check_passport(entry['passport_number']):
            illegal_keys.append('passport_number')
        elif not self.check_height(entry['height']):
            illegal_keys.append('height')
        elif not self.check_age(entry['age']):
            illegal_keys.append('age')
        elif not self.check_address(entry['address']):
            illegal_keys.append('address')
        elif not self.check_university(entry['university']):
            illegal_keys.append('university')
        elif not self.check_degree(entry['academic_degree']):
            illegal_keys.append('academic_degree')
        elif not self.check_worldview(entry['worldview']):
            illegal_keys.append('worldview')

        return illegal_keys

    def check_telephone(self, email: str) -> bool:

        pattern = "\+[0-9]-\([0-9]{3}\)\-[0-9]{3}\-[0-9]{2}\-[0-9]{2}"
        if re.match(pattern, email):
            return True
        return False

    def check_inn(self, inn: str) -> bool:

        pattern = '^\\d{12}$'

        if re.match(pattern, inn):
            return True
        return False

    def check_passport(self, passport: int) -> bool:
        return len(str(passport)) == 6

    def check_height(self, height: str) -> bool:
        try:
            float_height = float(height)
            return 2.2 > float_height > 1.2
        except ValueError:
            return False

        return True

    def check_age(self, age: str) -> bool:

        try:
            int_age = int(age)
        except ValueError:
            return False

        return int_age >= 18 and int_age < 110

    def check_address(self, address: str) -> bool:

        pattern = ".+[0-9]+"

        if re.match(pattern, address):
            return True
        return False

    def check_university(self, university: str) -> bool:

        pattern = "[а-яА-Я]+"

        if re.match(pattern, university):
            return True
        return False

    def check_degree(self, degree: str) -> bool:

        pattern = "[a-zA-Zа-яА-Я]+"

        if re.match(pattern, degree):
            return True
        return False

    def check_worldview(self, worldview: str) -> bool:

        pattern = "[a-zA-Zа-яА-Я]+"

        if re.match(pattern, worldview):
            return True
        return False
