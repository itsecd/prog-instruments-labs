import read_write
from checksum import calculate_checksum
from pattern import find_invalid_rows


def main():
    try:
        settings = read_write.read_json('settings.json')
        df = read_write.read_csv(settings["data"])

        pattern = read_write.read_json(settings["patterns"])

        index = find_invalid_rows(pattern, df)
        checksum = calculate_checksum(index)
        result = {"variant": int(settings['var']), "checksum": checksum}
        read_write.write_json(settings['result'], result)

    except Exception as e:
        print(f"Ошибка в main: {e}")


if __name__ == "__main__":
    main()
