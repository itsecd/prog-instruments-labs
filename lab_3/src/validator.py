import re

from files_handler import get_csv_path, load_csv


def main():
    print(load_csv(get_csv_path()))


if __name__ == "__main__":
    main()