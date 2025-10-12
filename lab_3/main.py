from utils.checksum import calculate_checksum
from utils.file_manager import read_csv, read_json, write_json
from utils.pattern_library import find_error_rows


def main():
    try:
        settings = read_json('settings.json')
        df = read_csv(settings['data'])
        patterns_dict = read_json(settings['patterns'])
        error_rows = find_error_rows(df, patterns_dict)
        checksum = calculate_checksum(error_rows)

        result_data = {
            "variant": int(settings['var']),
            "checksum": checksum
        }

        write_json(settings['result'], result_data)

    except FileNotFoundError as e:
        print(f" Файл не найден: {e}")
    except Exception as e:
        print(f" Неизвестная ошибка: {e}")


if __name__ == "__main__":
    main()