import json
import re
import pandas as pd

from checksum import calculate_checksum, serialize_result


class FileProcessor:
    def __init__(self, settings_path: str = "settings.json"):
        """
        Class initialization
        """
        with open(settings_path, 'r', encoding = 'utf-8') as file:
            self.settings = json.load(file)
        
        with open(self.settings['parser'], 'r', encoding = 'utf-8') as file:
            self.parsers = json.load(file)

        self.variant = self.settings['variant']
        self.csv_path = self.settings['input_file']


    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV-FILE
        :return: DataFrame
        """
        try:
            return pd.read_csv(self.csv_path)
        except Exception as e:
            raise RuntimeError(f"Error while loading CSV-file {e}")
        

    def validate_field(self, value: any, pattern: str) -> bool:
        """
        Validation 1 field by regular expression
        :param value: value of string
        :param pattern: regular expression
        :return: validation flag
        """
        if pd.isna(value):
            return False

        try:
            return bool(re.match(pattern, str(value)))
        except Exception:
            return False
        

    def validate_data(self, df: pd.DataFrame) -> list[int]:
        """
        Data validation
        :param df: DataFrame
        :return: Indices of invalid srtings
        """
        invalid_indices = []
        
        for idx, row in df.iterrows():
            is_valid = True
            for col, pattern in self.parsers.items():
                if col in df.columns:
                    if not self.validate_field(row[col], pattern):
                        is_valid = False
                        break
            
            if not is_valid:
                invalid_indices.append(idx)
        
        return invalid_indices
    

    def process_data(self) -> None:
        """Основной процесс обработки данных"""
        print(f"Обработка данных для варианта {self.variant}")
        print(f"CSV файл: {self.csv_path}")
        
        df = self.load_data()
        if df.empty:
            print("Не удалось загрузить данные")
            return
        
        print(f"Загружено строк: {len(df)}")
        print(f"Столбцы: {list(df.columns)}")
        
        invalid_indices = self.validate_data(df)
        print(f"Найдено невалидных строк: {len(invalid_indices)}")
        
        checksum = calculate_checksum(invalid_indices)
        print(f"Контрольная сумма: {checksum}")
        
        serialize_result(self.variant, checksum, self.settings['result'])
        print(f"Результат сохранен в {self.settings['result']}")


def main():
    """Основная функция"""
    validator = FileProcessor()
    validator.process_data()

if __name__ == "__main__":
    main()