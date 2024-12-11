import os

import nist_test

from const import DIR, FILE


if __name__ == "__main__":
    java_seq = nist_test.load(os.path.join(DIR,FILE))["java"]
    cpp_seq = nist_test.load(os.path.join(DIR,FILE))["cpp"]
    print("Nist java test:")
    print(nist_test.frequency_bit_test(java_seq))
    print(nist_test.identical_consecutive_bits(java_seq))
    print(nist_test.longest_sequence_of_ones_test(java_seq))
    print("Nist cpp test:")
    print(nist_test.frequency_bit_test(cpp_seq))
    print(nist_test.identical_consecutive_bits(cpp_seq))
    print(nist_test.longest_sequence_of_ones_test(cpp_seq))
    