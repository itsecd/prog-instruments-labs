from filereading import read_sequence
from seq_analyze import analyze_sequence


def main():
    file = open("statistics.txt", "w+")
    file.close()

    cpp_vect = read_sequence("generators/cpp/CppGen.txt")
    analyze_sequence(cpp_vect, "C++")

    java_vect = read_sequence("generators/java/JavaGenerated.txt")
    analyze_sequence(java_vect, "Java")

    python_vect = read_sequence("generators/python/PyGen.txt")
    analyze_sequence(python_vect, "Python")


if __name__ == "__main__":
    main()
