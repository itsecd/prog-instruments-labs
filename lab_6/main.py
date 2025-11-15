from fileinput import *
from NIST_tests.frequency import *
from NIST_tests.consecutive import *
from NIST_tests.longest_units import *


def main():
    """
    This function implements NIST tests for randomly grouped sequences.
    :return: None
    """
    try:
        settings = json_file_open("settings.json")
        sequence = {
            'cpp': file_open(settings["cpp_sequence"]),
            'java': file_open(settings["java_sequence"])
        }
        results = {}

        for i, sequence in sequence.items():
            results[i] = {
                'frequency_test': freq_test(sequence),
                'consecutive_bit_test': consecutive_bit_test(sequence),
                'longest_units_test': longest_units_test(sequence, settings['pi_values'])
            }

        json_file_save(settings['result'], results)

    except Exception as e:
        print(f"Ошибка: {e}")

    return 0


if __name__ == "__main__":
    main()