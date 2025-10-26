import argparse
from FileReader import FileReader
from Validator import Validator


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.txt')
    parser.add_argument('--output', default='output.txt')

    return parser


parser = createParser()
namespace = parser.parse_args()
print(namespace)

inputPath = namespace.input
outputPath = namespace.output

file = FileReader(inputPath)
validator = Validator(file.getData())
print(validator.parse_invalid())

f = open(outputPath, 'w')
for i in validator.parse_valid():
    f.write(str(i) + '\n')
