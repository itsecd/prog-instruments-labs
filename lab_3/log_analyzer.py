import re
from collections import defaultdict


def analyze_access_log(log_file_path):
    # Регулярные выражения для обнаружения угроз
    patterns = {
        'sql_injection': r'.*?(\%27|\')(?:\s*?(?:union|select|insert|drop|update|delete)|(?:\/\*.*?\*\/)).*?',
        'xss_attack': r'.*?(<script|javascript:|onerror=).*?',
        'path_traversal': r'.*?(\.\.\/|\.\.\\).*?',
        'sensitive_paths': r'.*?(\/admin|\/phpmyadmin|\/wp-login|\.env|\.git).*?',
        'suspicious_user_agent': r'.*?(nmap|sqlmap|wget|curl|python-requests).*?',
        'ip_blacklist': r'(?:10\.|192\.168|172\.(?:1[6-9]|2[0-9]|3[0-1]))'  # Пример: внутренние IP
    }

    threats = defaultdict(list)

    with open(log_file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            for threat_type, pattern in patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    threats[threat_type].append((line_num, line.strip()))

    return threats


def main():
    log_file = 'access.log'
    results = analyze_access_log(log_file)

    for threat, entries in results.items():
        print(f"Обнаружена угроза: {threat}")
        for line_num, line in entries:
            print(f"Строка {line_num}: {line}")
        print("-" * 50)


if __name__ == '__main__':
    main()