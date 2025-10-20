from TESTS.tests import (
    frequency_bitwise_test,
    identical_consecutive_bits,
    longest_tes_sequence,
)
from io_operations import read_json, read_file, write_json


def main():
    try:
        config = read_json("settings.json")
        pi = config["pi_i"]
        block_size = config["block_size"]

        cpp_seq = read_file(config["seq_CPP"])
        java_seq = read_file(config["seq_JAVA"])
        results = {
            "C++": {
                "frequency_bitwise_test": frequency_bitwise_test(cpp_seq),
                "identical_consecutive_bits": identical_consecutive_bits(cpp_seq),
                "longest_tes_sequence": longest_tes_sequence(cpp_seq, pi, block_size),
            },
            "Java": {
                "frequency_bitwise_test": frequency_bitwise_test(java_seq),
                "identical_consecutive_bits": identical_consecutive_bits(java_seq),
                "longest_tes_sequence": longest_tes_sequence(java_seq, pi, block_size),
            },
        }

        write_json(config["test_results"], results)

    except Exception as e:
        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    main()
