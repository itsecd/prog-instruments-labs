from filehandler.filehandler import read_data, save_data, read_csv


def main():
    try:
        table = read_csv("data/30.csv")
        print(table)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()