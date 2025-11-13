import checksum
import file_reader
import validator


def main():
    """Основная функция приложения"""
    try:
        lines = file_reader.read_csv_file('32.csv')

        invalid_rows = validator.process_csv_data(lines)

        control_sum = checksum.calculate_checksum(invalid_rows)

        checksum.serialize_result(32, control_sum)

        print(f"Найдено невалидных строк: {len(invalid_rows)}")
        print(f"Контрольная сумма: {control_sum}")

    except Exception as e:
        print(f"Ошибка выполнения: {e}")


if __name__ == "__main__":
    main()
