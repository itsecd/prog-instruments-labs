from checksum import calculate_checksum, serialize_result
from io_operations import read_csv, read_json, write_json
from pattern_fun import pattern_fun


def main():
    try:

        settings = read_json("settings.json")

        df = read_csv(settings["csv"])
        patterns = read_json(settings["patterns"])

        indexes = pattern_fun(patterns, df)

        checksum = calculate_checksum(indexes)
        result = serialize_result(settings["variant"], checksum)

        write_json(settings["result"], result)

    except Exception as e:
        print(e)


if __name__ == "__main__":

    main()
