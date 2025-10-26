import argparse
from FileReader import FileReader
from Validator import Validator


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.txt')
    parser.add_argument('--output', default='output.txt')

    return parser


parser = create_parser()
namespace = parser.parse_args()
print(namespace)

inputPath = namespace.input
outputPath = namespace.output

file = FileReader(inputPath)
validator = Validator(file.get_data())
print(validator.parse_invalid())

f = open(outputPath, 'w')
for i in validator.parse_valid():
    f.write(str(i) + '\n')
