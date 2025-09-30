import csv
import json
import re


from checksum import calculate_checksum, serialize_result


class FileProcessor:
    def __init__(self):
        """
        Class initialization
        """
        with open("settings.json", 'r', encoding = 'utf-8') as file:
            self.settings = json.load(file)
        
        with open("parser.json", 'r', encoding = 'utf-8') as file:
            self.parsers = json.load(file)


    def load_data(self) -> list[dict[str, str]]:
        """
        Load data from CSV-FILE
        :return: list of dict-s where every one is CSV-string
        """
        try:
            with open(self.settings['input_file'], 'r', encoding='utf-8') as file:
                return list(csv.DictReader(file))
        except Exception as e:
            raise RuntimeError(f"Error while loading CSV-file {e}")
        

    def validate_field(self, value: any, pattern: str) -> bool:
        """
        Validation 1 field by regular expression
        :param value: value of string
        :param pattern: regular expression
        :return: validation flag
        """
        if not value:
            return False

        return bool(re.match(pattern, str(value)))
        

    def validate_data(self, row: dict[str, str]) -> bool:
        """
        Data validation(rows)
        :param row: dict with data
        :return: flag if all of raws are valid
        """
        for field, pattern in self.parsers.items():
            if not self.validate_field(row.get(field), pattern):
                return False
        
        return True
    

    def process(self) -> None:
        """
        Data processing
        """
        data = self.load_data()

        invalid_indices = []
        for i, row in enumerate(data):
            if not self.validate_data(row):
                invalid_indices.append(i)

        checksum = calculate_checksum(invalid_indices)
        serialize_result(self.settings['variant'], checksum, self.settings['result'])
