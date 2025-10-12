import datetime as dt
import json
import math
import os
import random
from collections import defaultdict as dd


class MyClass:

    def __init__(self, name, value=0):
        self.NAME = name
        self.Value = value
        self.data_list = []
        self.data_dict = {}

    def add_data(self, *args):
        for item in args:
            self.data_list.append(item)

    def calculate_something(self, x, y):
        result = 0
        for i in range(len(self.data_list)):
            result += self.data_list[i] * x * y + math.sqrt(abs(self.Value)) if self.Value != 0 else 1
        return result

    def get_name(self):
        return self.NAME


class AnotherClass(MyClass):

    def __init__(self, name, value=0, extra_param=None):
        super().__init__(name, value)
        self.EXTRA = extra_param
        self.timestamp = dt.datetime.now()
        self._internal_counter = 0

    def process_data(self):
        if len(self.data_list) == 0:
            return None

        temp_result = 0
        for idx in range(len(self.data_list)):
            temp_result += self.data_list[idx] ** 2 + math.log(idx + 1) if idx > 0 else 0

        self._internal_counter += 1
        return temp_result / len(self.data_list) if len(self.data_list) > 0 else 0

    def complicated_method(self, param1, param2):
        if param1 > param2:
            result = []
            for i in range(param1):
                if i % 2 == 0:
                    result.append(i * param2 + self.Value)
                elif i % 3 == 0:
                    result.append(i * param1 - self.Value)
                else:
                    result.append(0)
            return result
        elif param1 < param2:
            return {'data': self.data_list, 'extra': self.EXTRA, 'count': self._internal_counter}
        else:
            return None


def global_function_one(a, b, c=None, d=None):
    if c is None:
        c = []

    result = []
    for i in range(a):
        for j in range(b):
            if i != j:
                value = i * j + sum(d) - len(c)
                result.append(value)

    if len(result) > 0:
        return max(result), min(result), sum(result) / len(result)
    else:
        return 0, 0, 0


def global_function_two(data, threshold=100):
    if not data:
        return []

    output = []
    for item in data:
        if isinstance(item, int) or isinstance(item, float):
            if item > threshold:
                output.append(item * 2)
            else:
                output.append(item / 2 if item != 0 else 0)
        elif isinstance(item, str):
            output.append(item.upper() + "_processed")
        else:
            output.append(None)

    return output


def create_objects_and_use_them():
    obj1 = MyClass("FirstObject", 10)
    obj1.add_data(1, 2, 3, 4, 5)

    obj2 = AnotherClass("SecondObject", 20, "extra_value")
    obj2.add_data(10, 20, 30)

    result1 = obj1.calculate_something(2, 3)
    result2 = obj2.process_data()
    result3 = obj2.complicated_method(5, 3)

    return {'obj1_result': result1, 'obj2_result': result2, 'obj3_result': result3}


def handle_files(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()

        data = json.loads(content)

        processed_data = []
        for key, value in data.items():
            if isinstance(value, list):
                processed_data.extend(value)
            elif isinstance(value, dict):
                for k, v in value.items():
                    processed_data.append(str(k) + ":" + str(v))
            else:
                processed_data.append(value)

        return processed_data

    except Exception as e:
        print(f"Error: {e}")
        return []


def generate_random_data(count):
    data = []
    for i in range(count):
        if i % 2 == 0:
            data.append(random.randint(1, 100))
        elif i % 3 == 0:
            data.append(random.random() * 100)
        else:
            data.append(f"string_{i}")

    return data


def complex_nested_function(n):
    if n <= 0:
        return []

    result = []
    for i in range(n):
        if i == 0:
            result.append(1)
        else:
            temp = []
            for j in range(i):
                if j == 0 or j == i - 1:
                    temp.append(1)
                else:
                    temp.append(result[i - 1][j - 1] + result[i - 1][j] if i > 1 and j < len(result[i - 1]) else 0)
            result.append(temp)

    return result


def another_complex_function(data):
    if not data:
        return {}

    counter = dd(int)
    for item in data:
        if isinstance(item, int):
            if item % 2 == 0:
                counter['even'] += item
            else:
                counter['odd'] += item
        elif isinstance(item, str):
            counter['strings'] += len(item)
        elif isinstance(item, list):
            counter['lists'] += len(item)
        else:
            counter['other'] += 1

    result = {}
    for k, v in counter.items():
        if v > 0:
            result[k] = v

    return result


class UtilityClass:

    @staticmethod
    def string_operations(s):
        if not s:
            return ""

        result = ""
        for char in s:
            if char.isalpha():
                if char.isupper():
                    result += char.lower()
                else:
                    result += char.upper()
            elif char.isdigit():
                result += str(int(char) * 2)
            else:
                result += "*"

        return result

    @staticmethod
    def number_operations(nums):
        if not nums:
            return []

        positive = [x for x in nums if x > 0]
        negative = [x for x in nums if x < 0]
        zero_count = len([x for x in nums if x == 0])

        return {
            'positive_sum': sum(positive),
            'negative_sum': sum(negative),
            'positive_avg': sum(positive) / len(positive) if positive else 0,
            'negative_avg': sum(negative) / len(negative) if negative else 0,
            'zero_count': zero_count
        }


def main():
    print("Starting program...")

    # Generate some random data
    random_data = generate_random_data(50)
    print(f"Generated {len(random_data)} random items")

    # Process the data
    processed_data = global_function_two(random_data, 50)
    print(f"Processed data: {processed_data[:10]}...")

    # Create objects and use them
    object_results = create_objects_and_use_them()
    print(f"Object results: {object_results}")

    # Complex nested function
    triangle = complex_nested_function(6)
    print(f"Complex nested result: {triangle}")

    # Another complex function
    stats = another_complex_function(random_data)
    print(f"Statistics: {stats}")

    # Utility class operations
    test_string = "Hello123 World!"
    modified_string = UtilityClass.string_operations(test_string)
    print(f"Modified string: {modified_string}")

    numbers = [1, -2, 3, -4, 0, 5, 0]
    number_stats = UtilityClass.number_operations(numbers)
    print(f"Number statistics: {number_stats}")

    # File operations (if file exists)
    if os.path.exists('data.json'):
        file_data = handle_files('data.json')
        print(f"File data: {file_data[:5]}...")
    else:
        print("No data file found")

    # Global function one
    max_val, min_val, avg_val = global_function_one(5, 5)
    print(f"Global function results: max={max_val}, min={min_val}, avg={avg_val}")

    # More object operations
    obj = AnotherClass("TestObject", 42, "additional_info")
    obj.add_data(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    for i in range(3):
        result = obj.process_data()
        print(f"Processing {i + 1}: {result}")

    complex_result = obj.complicated_method(10, 5)
    print(f"Complex method result: {complex_result}")

    # Final calculations
    final_data = []
    for i in range(100):
        if i % 7 == 0:
            final_data.append(i * 2)
        elif i % 5 == 0:
            final_data.append(i * 3)
        else:
            final_data.append(i)

    processed_final = global_function_two(final_data, 50)
    stats_final = another_complex_function(processed_final)

    print(f"Final statistics: {stats_final}")
    print("Program completed!")


if __name__ == "__main__":
    main()
