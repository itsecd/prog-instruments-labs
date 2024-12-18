from utility.functions import write_text, read_json
from utility.tests import consecutive_bits_test, longest_sequence_test, frequency_bitwise_test

if __name__ == "__main__":
    results = "lab_2/NIST_results.txt"
    sequences = read_json("lab_2/gen/gen_results.json")
    cpp = sequences["cpp"]
    java = sequences["java"]
    
    write_text(results, str(frequency_bitwise_test(cpp)) + " C++ frequency test\n")
    write_text(results, str(frequency_bitwise_test(java)) + " Java frequency test\n")
    write_text(results, str(consecutive_bits_test(cpp)) + " C++ cons. bits test\n")
    write_text(results, str(consecutive_bits_test(java)) + " Java cons. bits test\n")
    write_text(results, str(longest_sequence_test(cpp)) + " C++ longest seq. test\n")
    write_text(results, str(longest_sequence_test(java)) + " Java longest seq. test\n")