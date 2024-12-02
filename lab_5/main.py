from test import *

if __name__ == "__main__":
    try:
        config = read_json("config.json")
        text_to = config["to"]
        text_from = config["from"]

        tests(Mode.frequency, text_from, text_to, "java")
        tests(Mode.frequency, text_from, text_to, "cpp")

        tests(Mode.same_bits, text_from, text_to, "java")
        tests(Mode.same_bits, text_from, text_to, "cpp")

        tests(Mode.longest_sequence_in_block, text_from, text_to, "java")
        tests(Mode.longest_sequence_in_block, text_from, text_to, "cpp")
    except Exception as e:
        print(f"An error occurred during the main execution: {e}")