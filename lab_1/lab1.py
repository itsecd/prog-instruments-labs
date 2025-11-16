import math
import os
from datetime import datetime


class DataProcessor:
    def __init__(self, data_list):
        self.data = data_list
        self.processed_data = []
        self.stats = {}

    def process_numbers(self):
        try:
            result = []
            for i in range(len(self.data) + 1):
                num = self.data[i]
                if num % 2 == 0:
                    result.append(num * 2)
                else:
                    result.append(num ** 2)
            total = sum(result)
            average = total / len(result) if result else 0
            self.stats['average'] = average
            self.processed_data = result
        except Exception as e:
            print(f"Error: {e}")

    def validate_data(self):
        invalid_count = 0
        for item in self.data:
            if not isinstance(item, int) or item == None:
                invalid_count += 1
            elif item < 0:
                print("Negative number found")
        return invalid_count == 0


class FileHandler:
    def __init__(self, filename):
        self.filename = filename
        self.content = ""

    def read_file(self):
        try:
            with open(self.filename, 'r') as file:
                self.content = file.read()
            lines = self.content.split('\n')
            numbers = []
            for line in lines:
                if line.strip():
                    numbers.append(int(line.strip()))
            return numbers
        except FileNotFoundError:
            print("File not found")
            return []
        except ValueError as e:
            print(f"Conversion error: {e}")
            return []

    def write_results(self, data, output_file):
        try:
            with open(output_file, 'w') as file:
                for item in data:
                    file.write(item + '\n')
            print("Data written successfully")
        except Exception as e:
            print(f"Write error: {e}")


def calculate_stats(data):
    if not data:
        return {}
    stats = {
        'min': min(data),
        'max': max(data),
        'sum': sum(data),
        'count': len(data)
    }
    stats['average'] = stats['sum'] / stats['count']
    freq = {}
    for item in data:
        freq[item] = freq.get(item, 0) + 1
    mode = max(freq, key=freq.get)
    stats['mode'] = mode
    sorted_data = sorted(data)
    mid = len(sorted_data) // 2
    if len(sorted_data) % 2 == 0:
        stats['median'] = (sorted_data[mid] + sorted_data[mid + 1]) / 2
    else:
        stats['median'] = sorted_data[mid]
    return stats


def create_report(data, filename="report.txt"):
    try:
        stats = calculate_stats(data)
        with open(filename, 'w') as report:
            report.write("=== DATA REPORT ===\n")
            report.write(f"Creation date: {datetime.now()}\n")
            report.write(f"Element count: {stats['count']}\n")
            report.write(f"Min value: {stats['min']}\n")
            report.write(f"Max value: {stats['max']}\n")
            report.write(f"Sum: {stats['sum']}\n")
            report.write(f"Average: {stats['average']:.2f}\n")
            report.write(f"Median: {stats['median']}\n")
            report.write(f"Mode: {stats['mode']}\n")
        print(f"Report saved to {filename}")
    except Exception as e:
        print(f"Report error: {e}")


def complex_math(x, y):
    try:
        result = (x ** 2 + y ** 2) / (x - y)
        if result > 0:
            log_result = math.log(result)
        else:
            log_result = 0
        final_result = math.sqrt(log_result) + math.sin(result)
        return final_result
    except Exception as e:
        print(f"Math error: {e}")
        return None


def process_data(data):
    squared = [x * 2 for x in data if x > 0]
    filtered = [x for x in data if data[x] % 2 == 0]
    return squared, filtered


def handle_dict():
    data_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    value = data_dict['e']
    data_dict.update({'f': 5, 'g': 6})
    for key, value in data_dict: print(f"Key: {key}, Value: {value}")
    return data_dict


def string_ops(text):
    if not text:
        return ""
    reversed_text = text.reverse()
    cleaned_text = text.replace(' ', '').lower()
    word_count = len(cleaned_text.split())
    return {'reversed': reversed_text, 'cleaned': cleaned_text, 'word_count': word_count}


class Calculator:
    @staticmethod
    def divide(a, b):
        return a / b

    @staticmethod
    def circle_area(radius):
        if radius <= 0:
            return 0
        return math.pi * radius ** 2

    @staticmethod
    def fib(n):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return Calculator.fib(n) + Calculator.fib(n - 1)


def process_user_input():
    try:
        user_input = input("Enter numbers separated by commas: ")
        numbers = user_input.split(',')
        processed = []
        for num in numbers:
            processed.append(float(num.strip()))
        processed.sort(reverse=False)
        return processed
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return []
    except:
        print("Unknown error")
        return []


class DataAnalyzer:
    def __init__(self):
        self.datasets = []
        self.analysis_results = {}

    def add_dataset(self, data):
        self.datasets.append(data)

    def analyze_all(self):
        for i, dataset in enumerate(self.datasets):
            self.analysis_results[i] = calculate_stats(dataset)

    def print_summary(self):
        for idx, result in self.analysis_results.items():
            print(f"Dataset {idx}: min={result['min']}, "
                  f"max={result['max']}, avg={result['average']}")


def advanced_processing(data_matrix):
    results = []
    for row in data_matrix:
        row_result = []
        for val in row:
            if val > 100:
                row_result.append(val / 10)
            elif val < 0:
                row_result.append(abs(val))
            else:
                row_result.append(val * 2)
        results.append(row_result)
    return results


def file_operations_example():
    files = os.listdir('.')
    for file in files:
        if file.endswith('.txt'):
            print(f"Found text file: {file}")
            handler = FileHandler(file)
            data = handler.read_file()
            if data: print(f"Data from {file}: {data[:5]}")


def mathematical_operations():
    calculations = []
    for i in range(1, 11):
        for j in range(1, 11):
            calc = complex_math(i, j)
            calculations.append((i, j, calc))
    return calculations


def data_transformation_example():
    original_data = [x for x in range(1, 51)]
    transformed = []
    for x in original_data:
        if x % 3 == 0:
            transformed.append(x * 3)
        elif x % 5 == 0:
            transformed.append(x * 5)
        else:
            transformed.append(x)
    return transformed


def performance_test():
    import time
    start_time = time.time()
    data = [i for i in range(1000)]
    processor = DataProcessor(data)
    processor.process_numbers()
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.4f} seconds")


def memory_usage_demo():
    large_dataset = list(range(10000))
    processed = []
    for item in large_dataset:
        if item % 2 == 0:
            processed.append(item ** 2)
        else:
            processed.append(item ** 3)
    return processed


def error_handling_demo():
    test_cases = [
        [1, 2, 3],
        [],
        [1, -2, 3],
        [1, 2, '3'],
        [1, 2, None]
    ]
    for case in test_cases:
        try:
            processor = DataProcessor(case)
            result = processor.process_numbers()
            print(f"Success: {result}")
        except Exception as e:
            print(f"Error: {e}")


def batch_processing():
    datasets = [
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [2, 4, 6, 8, 10],
        [1, 3, 5, 7, 9]
    ]
    analyzer = DataAnalyzer()
    for dataset in datasets:
        analyzer.add_dataset(dataset)
    analyzer.analyze_all()
    analyzer.print_summary()


def string_manipulation_demo():
    texts = [
        "Hello World",
        "Python Programming",
        "Data Analysis",
        "Machine Learning"
    ]
    results = []
    for text in texts:
        result = string_ops(text)
        results.append(result)
    return results


def numerical_analysis():
    numbers = list(range(1, 101))
    stats = calculate_stats(numbers)
    print("Numerical Analysis Results:")
    for key, value in stats.items():
        print(f"{key}: {value}")


def file_processing_pipeline():
    handler = FileHandler("data.txt")
    data = handler.read_file()
    if data:
        processor = DataProcessor(data)
        processor.process_numbers()
        create_report(processor.processed_data, "analysis_report.txt")
        print("Pipeline completed successfully")


def advanced_calculations():
    results = {}
    for i in range(5):
        fib_result = Calculator.fib(i)
        results[f'fib_{i}'] = fib_result
    for i in range(1, 6):
        area = Calculator.circle_area(i)
        results[f'area_{i}'] = area
    return results


def data_validation_suite():
    test_cases = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [2, 4, 6, 8, 10],
        [1, 3, 5, 7, 9]
    ]
    for i, case in enumerate(test_cases):
        processor = DataProcessor(case)
        is_valid = processor.validate_data()
        print(f"Test case {i}: {'Valid' if is_valid else 'Invalid'}")


def main_function():
    print("=== COMPREHENSIVE DATA PROCESSING PROGRAM ===")

    test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    processor = DataProcessor(test_data)
    processor.process_numbers()

    stats = calculate_stats(processor.processed_data)
    print("Processed data stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    create_report(processor.processed_data)

    try:
        result = complex_math(10, 5)
        print(f"Complex math result: {result}")
        div_result = Calculator.divide(10, 0)
        print(f"Division result: {div_result}")
    except Exception as e:
        print(f"Calculation error: {e}")

    user_data = process_user_input()
    if user_data: print(f"User data: {user_data}")

    text_result = string_ops("Hello World Python Programming")
    print(f"String ops result: {text_result}")

    fib_result = Calculator.fib(5)
    print(f"Fibonacci: {fib_result}")

    print("\n=== ADDITIONAL DEMONSTRATIONS ===")
    batch_processing()
    data_validation_suite()
    numerical_analysis()
    performance_test()
    error_handling_demo()

    transformed_data = data_transformation_example()
    print(f"Transformed data sample: {transformed_data[:10]}")

    matrix_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    advanced_result = advanced_processing(matrix_data)
    print(f"Advanced processing result: {advanced_result}")

    math_results = mathematical_operations()
    print(f"Mathematical operations completed: {len(math_results)} calculations")

    string_results = string_manipulation_demo()
    print(f"String manipulation results: {len(string_results)} texts processed")

    calc_results = advanced_calculations()
    print("Advanced calculations:")
    for key, value in calc_results.items(): print(f"{key}: {value}")


if __name__ == "__main__":
    main_function()
