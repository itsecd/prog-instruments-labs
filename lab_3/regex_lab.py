import argparse
import csv
from checksum import calculate_checksum, serialize_result
from patterns import REGEXES


def read_csv(filename: str) -> list[dict]:
  """
  Read .csv file.

  Args:
    filename: path to .csv file

  Returns:
    list of dictionaries (one dictionary is one row from .csv file)
  """
  with open(filename, mode = "r", encoding = "utf-16") as file:
    reader = csv.DictReader(file, delimiter = ";")
    return list(reader)


def find_invalid_rows(rows_list: list[dict], REGEXES: dict) -> list[int]:
  """
  Checks the list of dictionaries (strings from .csv file) for matching their values to regular expressions.

  Args:
    rows_list: list of dictionaries (one dictionary is one row from .csv file)
    REGEXES: dictionary of regular expressions for each column

  Returns:
    invalid_rows: list of row indexes, where value does not match regular expression
  """
  invalid_rows = []

  for index, row in enumerate(rows_list):
    for column_name, regex in REGEXES.items():
      row_value = row.get(column_name)
      
      if not regex.match(row_value):
        invalid_rows.append(index)
        break

  return invalid_rows


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("-csv", '--csv_file_name', type = str, default = '27.csv', help = 'name of csv file')
  parser.add_argument("-r", '--result_file_name', type = str, default = 'result.json', help = 'name of file with result')
  parser.add_argument("-v", '--variant', type = str, default = 27, help='variant number')
  args = parser.parse_args()
  
  print(f"The name of csv file is: {args.csv_file_name}")
  print(f"The name of file with result is: {args.result_file_name}")
  print(f"The variant number is: {args.variant}")

  try:
    rows_list = read_csv(args.csv_file_name)
    invalid_rows = find_invalid_rows(rows_list, REGEXES)
    value_checksum = calculate_checksum(invalid_rows)
    serialize_result(args.result_file_name, args.variant, value_checksum)

  except Exception as error:
    print(f"Error: {error}")


if __name__ == "__main__":
  main()