import re


def read_log_file(file_path: str):
    try:
        with open(file_path, 'r') as file:
            return file.readlines()
    except FileNotFoundError:
        print(f'Error: file {file_path} is not found')
        return []

  
def parse_log_line(line: str):
    log_pattern = (
        r'^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) - - '
        r'\[(\d{1,2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4})\] '
        r'"(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|CONNECT|TRACE) '
        r'([^"]*) '  
        r'HTTP/(1\.0|1\.1|2\.0)" '
        r'(1\d{2}|2\d{2}|3\d{2}|4\d{2}|5\d{2}) '
        r'(\d+)$' 
    )

    match = re.match(log_pattern, line.strip())
    if match:
        return {
            'ip': match.group(1),
            'time': match.group(2),
            'method': match.group(3),
            'url': match.group(4),
            'status': int(match.group(6)),
            'size' : int(match.group(7))
        }
    return None


def main():
    file_path: str = 'data.log'
    file_lines = read_log_file(file_path)

    print(parse_log_line(file_lines[0]))
    print(file_lines[0])


if __name__ == '__main__':
    main()