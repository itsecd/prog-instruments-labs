import datetime
import json
import math
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple


class DataProcessor:
    def __init__(self, data_source: str):
        self.data_source = data_source
        self.processed_data = []
        self._cache = {}

    def load_data(self, file_path: str) -> List[Dict]:
        data_list = []
        try:
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
                for index, item in enumerate(raw_data):
                    if self._validate_item(item):
                        transformed = self._transform_item(item, index)
                        data_list.append(transformed)
                        print(f"Processed item {index}: {transformed}")
                    else:
                        print(f"Invalid item at index {index}: {item}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except Exception as e:
            print(f"Unexpected error loading data: {e}")
            return []
        return data_list

    def _validate_item(self, item: Dict) -> bool:
        if not item or not isinstance(item, dict):
            return False
        if 'id' not in item or not isinstance(item['id'], (int, str)):
            return False
        if 'value' not in item:
            return False
        try:
            float(item['value'])
        except (ValueError, TypeError):
            return False
        return True

    def _transform_item(self, item: Dict, index: int) -> Dict:
        transformed = {}
        transformed['identifier'] = str(item['id'])
        transformed['numeric_value'] = float(item['value'])
        transformed['processing_index'] = index
        transformed['processed_at'] = datetime.datetime.now()

        if 'timestamp' in item:
            try:
                timestamp_format = '%Y-%m-%d %H:%M:%S'
                transformed['time'] = datetime.datetime.strptime(
                    item['timestamp'], timestamp_format
                )
            except ValueError:
                transformed['time'] = None

        if 'category' in item:
            transformed['category'] = item['category'].upper()

        return transformed

    def batch_process(self, items: List[Dict]) -> List[Dict]:
        processed_items = []
        for i, item in enumerate(items):
            if self._validate_item(item):
                processed = self._transform_item(item, i)
                processed_items.append(processed)
        return processed_items

    def clear_cache(self):
        self._cache = {}
        print("Cache cleared")

    def get_stats(self) -> Dict:
        return {
            'processed_count': len(self.processed_data),
            'cache_size': len(self._cache),
            'data_source': self.data_source
        }


class Calculator:
    def __init__(self, precision: int = 2):
        self.precision = precision
        self._calculation_history = []

    def calculate_average(self, numbers: List[float]) -> float:
        if not numbers:
            self._add_to_history('average', numbers, 0.0)
            return 0.0
        total = 0.0
        for n in numbers:
            total += n
        result = total / len(numbers)
        self._add_to_history('average', numbers, result)
        return round(result, self.precision)

    def calculate_standard_deviation(self, numbers: List[float]) -> float:
        if len(numbers) < 2:
            self._add_to_history('std_dev', numbers, 0.0)
            return 0.0
        avg = self.calculate_average(numbers)
        variance = 0.0
        for num in numbers:
            variance += (num - avg) ** 2
        result = math.sqrt(variance / (len(numbers) - 1))
        self._add_to_history('std_dev', numbers, result)
        return round(result, self.precision)

    def find_extremes(self, numbers: List[float]) -> Tuple[float, float]:
        if not numbers:
            self._add_to_history('extremes', numbers, (0.0, 0.0))
            return (0.0, 0.0)
        min_val = numbers[0]
        max_val = numbers[0]
        for num in numbers[1:]:
            if num < min_val:
                min_val = num
            if num > max_val:
                max_val = num
        result = (round(min_val, self.precision), round(max_val, self.precision))
        self._add_to_history('extremes', numbers, result)
        return result

    def calculate_median(self, numbers: List[float]) -> float:
        if not numbers:
            return 0.0
        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)
        if n % 2 == 1:
            result = sorted_numbers[n // 2]
        else:
            result = (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2
        self._add_to_history('median', numbers, result)
        return round(result, self.precision)

    def calculate_range(self, numbers: List[float]) -> float:
        if not numbers:
            return 0.0
        min_val, max_val = self.find_extremes(numbers)
        result = max_val - min_val
        self._add_to_history('range', numbers, result)
        return round(result, self.precision)

    def _add_to_history(self, operation: str, data: List[float], result: Any):
        entry = {
            'timestamp': datetime.datetime.now(),
            'operation': operation,
            'data': data,
            'result': result
        }
        self._calculation_history.append(entry)

    def get_history(self) -> List[Dict]:
        return self._calculation_history.copy()

    def clear_history(self):
        self._calculation_history = []
        print("Calculation history cleared")

    def get_statistics_summary(self, numbers: List[float]) -> Dict:
        return {
            'average': self.calculate_average(numbers),
            'std_dev': self.calculate_standard_deviation(numbers),
            'min': self.find_extremes(numbers)[0],
            'max': self.find_extremes(numbers)[1],
            'median': self.calculate_median(numbers),
            'range': self.calculate_range(numbers)
        }


def process_user_input(input_string: str,
                       validation_rules: Optional[Dict] = None
                       ) -> Dict[str, Any]:
    result = {'success': False, 'data': None, 'error': None, 'warnings': []}

    if not input_string or len(input_string.strip()) == 0:
        result['error'] = 'Empty input'
        return result

    cleaned_input = input_string.strip()

    if validation_rules and 'max_length' in validation_rules:
        if len(cleaned_input) > validation_rules['max_length']:
            max_len = validation_rules["max_length"]
            result['warnings'].append(
                f'Input exceeds maximum length of {max_len} characters'
            )

    if validation_rules and 'min_length' in validation_rules:
        if len(cleaned_input) < validation_rules['min_length']:
            min_len = validation_rules["min_length"]
            result['error'] = f'Input must be at least {min_len} characters long'
            return result

    cleaned_input = cleaned_input.lower()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if cleaned_input.isdigit():
        result['data'] = int(cleaned_input)
        result['success'] = True
    elif re.match(r'^[a-zA-Z\s]+$', cleaned_input):
        result['data'] = cleaned_input.upper()
        result['success'] = True

    elif re.match(email_pattern, cleaned_input):
        result['data'] = cleaned_input
        result['success'] = True
    else:
        result['error'] = 'Invalid input format'

    return result


def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    pattern = r'^\+?[1-9]\d{1,14}$'
    return bool(re.match(pattern, phone))


def sanitize_input(input_string: str) -> str:
    sanitized = input_string.strip()
    sanitized = sanitized.replace('\0', '').replace('\r', '').replace('\n', '')
    return sanitized


def format_output(data: Any, format_type: str = 'string') -> str:
    if format_type == 'json':
        return json.dumps(data, indent=2, default=str)
    elif format_type == 'csv':
        if isinstance(data, list):
            return '\n'.join([','.join(map(str, item.values())) for item in data])
        return str(data)
    else:
        return str(data)


class DatabaseHandler:
    def __init__(
            self,
            host: str,
            port: int,
            database: str,
            username: str,
            password: str,
            connection_timeout: float = 10.0
    ):
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': username,
            'password': password,
            'timeout': connection_timeout
        }
        self.is_connected = False
        self.connection = None
        self.connection_time = None
        self.query_count = 0

    def connect(self) -> bool:
        try:
            host = self.connection_params['host']
            port = self.connection_params['port']
            print(f"Connecting to {host}:{port}")
            time.sleep(0.1)
            self.is_connected = True
            self.connection = "mock_connection_object"
            self.connection_time = datetime.datetime.now()
            print("Connection established successfully")
            return True
        except ConnectionError as e:
            print(f"Connection error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    def execute_query(self, query: str,
                      params: Optional[Tuple] = None) -> List[Dict]:
        if not self.is_connected:
            print("Not connected to database")
            return []

        self.query_count += 1
        print(f"Executing query #{self.query_count}: {query}")
        if params:
            print(f"With parameters: {params}")

        time.sleep(0.05)

        if "SELECT" in query.upper():
            if "WHERE age >" in query:
                return [
                    {'id': 1, 'name': 'Test', 'value': 100},
                    {'id': 2, 'name': 'Test2', 'value': 200}
                ]
            else:
                return [
                    {'id': 1, 'name': 'Test', 'value': 100},
                    {'id': 2, 'name': 'Test2', 'value': 200}]
        elif "INSERT" in query.upper():
            return [{'affected_rows': 1}]
        elif "UPDATE" in query.upper():
            return [{'affected_rows': 1}]
        elif "DELETE" in query.upper():
            return [{'affected_rows': 1}]
        else:
            return []

    def disconnect(self):
        if self.is_connected:
            print("Disconnecting from database")
            self.is_connected = False
            self.connection = None
            connection_duration = datetime.datetime.now() - self.connection_time
            print(f"Connection duration: {connection_duration}")

    def get_stats(self) -> Dict:
        return {
            'query_count': self.query_count,
            'connection_time': self.connection_time,
            'is_connected': self.is_connected
        }

    def batch_execute(self, queries: List[str]) -> List[List[Dict]]:
        results = []
        for query in queries:
            results.append(self.execute_query(query))
        return results

    def test_connection(self) -> bool:
        try:
            print("Testing database connection...")
            return self.connect() and self.disconnect()
        except ConnectionError as e:
            print(f"Connection test failed: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during connection test: {e}")
            return False


def generate_report(data: List[Dict],
                    output_file: Optional[str] = None,
                    include_details: bool = False) -> str:
    if not data:
        return "No data to generate report"

    calc = Calculator()
    values = [item['numeric_value'] for item in data if 'numeric_value' in item]

    if not values:
        return "No numeric values found"

    avg = calc.calculate_average(values)
    std_dev = calc.calculate_standard_deviation(values)
    min_val, max_val = calc.find_extremes(values)

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("DATA ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Report generated at: {datetime.datetime.now()}")
    report_lines.append(f"Total items processed: {len(data)}")
    report_lines.append(f"Valid numeric values: {len(values)}")
    report_lines.append(f"Average value: {avg:.2f}")
    report_lines.append(f"Standard deviation: {std_dev:.2f}")
    report_lines.append(f"Minimum value: {min_val:.2f}")
    report_lines.append(f"Maximum value: {max_val:.2f}")
    report_lines.append(f"Value range: {max_val - min_val:.2f}")

    if include_details:
        report_lines.append("\nDETAILED DATA:")
        report_lines.append("-" * 40)
        for i, item in enumerate(data[:5]):
            report_lines.append(
                f"{i + 1}. ID: {item.get('identifier', 'N/A')}, "
                f"Value: {item.get('numeric_value', 'N/A')}"
            )
        if len(data) > 5:
            report_lines.append(f"... and {len(data) - 5} more items")

    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)

    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        except FileNotFoundError:
            print(f"Directory not found: {os.path.dirname(output_file)}")
        except Exception as e:
            print(f"Unexpected error saving report: {e}")

    return report_text


def create_sample_data(count: int = 15) -> List[Dict]:
    sample_data = []
    categories = ['A', 'B', 'C', 'D']

    for i in range(count):
        item = {
            'id': i + 1,
            'value': str(round(random.uniform(10.0, 100.0), 2)),
            'timestamp': f'2023-01-{(i % 30) + 1:02d} {random.randint(10, 23):02d}:00:00',
            'category': random.choice(categories)
        }
        sample_data.append(item)

    return sample_data


def demonstrate_features():
    print("=== DEMONSTRATION OF FEATURES ===")

    sample_data = create_sample_data()
    print(f"Created {len(sample_data)} sample data items")

    with open('temp_data.json', 'w') as f:
        json.dump(sample_data, f)

    processor = DataProcessor('file')
    data = processor.load_data('temp_data.json')

    if data:
        report = generate_report(data, 'report.txt', include_details=True)
        print("\nGenerated Report:")
        print(report)
    else:
        print("No data processed")

    calc = Calculator()
    test_numbers = [10.5, 20.3, 15.7, 25.1, 18.9]

    print(f"\nCalculator demo with numbers: {test_numbers}")
    print(f"Average: {calc.calculate_average(test_numbers)}")
    print(f"Std Dev: {calc.calculate_standard_deviation(test_numbers)}")
    print(f"Min/Max: {calc.find_extremes(test_numbers)}")

    print("\n=== USER INPUT PROCESSING ===")
    test_inputs = ['123', 'hello world', 'test@example.com', '123abc']

    for input_str in test_inputs:
        result = process_user_input(input_str)
        status = "✓" if result['success'] else "✗"
        print(f"{status} Input: '{input_str}' -> {result}")

    print("\n=== DATABASE OPERATIONS ===")
    db_handler = DatabaseHandler('localhost', 5432, 'mydb', 'user', 'password')
    if db_handler.connect():
        results = db_handler.execute_query("SELECT * FROM users WHERE age > %s", (25,))
        print(f"Query results: {len(results)} users found")
        db_handler.disconnect()

    if os.path.exists('temp_data.json'):
        os.remove('temp_data.json')

    print("\nApplication finished successfully")


if __name__ == "__main__":
    demonstrate_features()