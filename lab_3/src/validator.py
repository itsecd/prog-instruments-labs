import re

from files_handler import get_csv_path, load_csv


def validate_telephone(phone):
    pattern = r"^\+7-\((\d{3})\)-(\d{3})-(\d{2})-(\d{2})$"


def main():
    print(load_csv(get_csv_path()))


if __name__ == "__main__":
    main()