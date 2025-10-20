import re

from const import PATH_TRAVEL_PATTERNS, XSS_PATTERNS, SQL_INJECTION_PATTERNS, MALICIOUS_AGENT_PATTERNS


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
        r'(\d+) ' 
        r'"((?:https?://[^"]+|-))" '
        r'"([^"]*)"$'
    )

    match = re.match(log_pattern, line.strip())
    if match:
        return {
            'ip': match.group(1),
            'time': match.group(2),
            'method': match.group(3),
            'url': match.group(4),
            'status': int(match.group(6)),
            'size' : int(match.group(7)),
            'referer': match.group(8),
            'user_agent': match.group(9)
        }
    return None


def write_result(filename: str, threat_name: str, line):
    with open(filename, 'a') as file:
        file.write(f'[{threat_name}] {str(line)}\n')


def detect_anomaly(log_line, result_file: str):
    for pattern in PATH_TRAVEL_PATTERNS:
        if re.search(pattern, log_line['url'], re.IGNORECASE):
            write_result(result_file, 'Path Traveling', log_line)
            break

    for pattern in XSS_PATTERNS:
        if re.search(pattern, log_line['url'], re.IGNORECASE):
            write_result(result_file, 'XSS', log_line)
            break

    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, log_line['url'], re.IGNORECASE):
            write_result(result_file, 'SQL injection', log_line)
            break


def detect_malicious_agents(log_line, result_file):
    for tool, pattern in MALICIOUS_AGENT_PATTERNS.items():
        if re.search(pattern, log_line['user_agent'], re.IGNORECASE):
            write_result(result_file, f'Malicious agent: {tool}', log_line)
            break


def main():
    file_path: str = 'data.log'
    file_lines = read_log_file(file_path)

    for line in file_lines:
        parsed_line = parse_log_line(line)
        if parsed_line is not None:
            detect_anomaly(parsed_line, 'result.txt')
            detect_malicious_agents(parsed_line, 'result.txt')


if __name__ == '__main__':
    main()