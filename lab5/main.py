import logging

from io_operations import read_file, read_json, write_json
from TESTS.tests import (
    frequency_bitwise_test,
    identical_consecutive_bits,
    longest_tes_sequence,
)

from lab5.logger import logger

logger = logging.getLogger("main")


def main():
    logger.info("Starting sequence tests application")
    try:
        logger.debug("Reading configuration")
        config = read_json("settings.json")
        pi = config["pi_i"]
        block_size = config["block_size"]
        logger.info(f"Configuration loaded: block_size={block_size}")

        logger.info("Reading CPP sequence")
        cpp_seq = read_file(config["seq_CPP"])
        logger.info("Reading JAVA sequence")
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
        logger.info("Saving test results")
        write_json(config["test_results"], results)

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    main()
