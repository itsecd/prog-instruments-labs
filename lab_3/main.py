import file_reader


def main():
    encoding = file_reader.detect_encoding('32.csv')
    print(f"Определенная кодировка файла: {encoding}")


if __name__ == "__main__":
    main()