import argparse

from filehandler.filehandler import read_data, save_data, read_csv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_file", type=str, help="directory to settings file")
    return parser.parse_args()


def main():
    try:
        settings = read_data(get_args().settings_file)
        table = read_csv(settings["csv_file"])
        print(table)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()