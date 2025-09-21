import re
from collections import defaultdict



class AccessLogAnalyzer:
    def __init__(self):
        # Регулярное выражение для парсинга строк access.log
        self.log_pattern = re.compile(
            r'(\d+\.\d+\.\d+\.\d+)\s-\s-\s\[(.*?)\]\s"(.*?)"\s(\d+)\s(\d+)\s"(.*?)"\s"(.*?)"'
        )
        # Паттерны для обнаружения различных типов атак
        self.patterns = {
            'sql_injection': [
                r'union\s+select',
                r'select.*from',
                r'insert\s+into',
                r'drop\s+table',
                r'or\s+1=1',
                r';\s*(--|#)',
                r'exec\(|execute\(|xp_cmdshell',
                r'waitfor\s+delay',
                r'benchmark\(|sleep\(',
                r'\b(sqlmap|sqli)\b'
            ],
            'xss_attack': [
                r'<script>',
                r'javascript:',
                r'onerror=|onload=|onmouseover=',
                r'alert\(|confirm\(|prompt\(',
                r'document\.cookie|window\.location',
                r'eval\(|setTimeout\(|setInterval\('
            ],
            'path_traversal': [
                r'\.\./|\.\.\\',
                r'etc/passwd',
                r'proc/self',
                r'win\.ini|boot\.ini',
                r'\.\.%2f|\.\.%5c'
            ],
            'suspicious_user_agent': [
                r'nmap|sqlmap|metasploit',
                r'nikto|wpscan|dirb',
                r'hydra|medusa|burpsuite',
                r'python-requests|curl|wget',
                r'zgrab|masscan|zmap',
                r'^$|^-$|unknown|undefined'  # Пустые или неопределенные UA
            ]

        }


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