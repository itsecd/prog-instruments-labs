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


def analyze_log(self, log_file_path):
    """Анализ всего файла лога"""
    suspicious_servers = defaultdict(list)
    ip_stats = defaultdict(lambda: {'requests': 0, 'errors': 0, 'threats': 0})

    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line_num, line in enumerate(file, 1):
            log_entry = self.parse_log_line(line.strip())
            if not log_entry:
                continue

            ip = log_entry['ip']
            ip_stats[ip]['requests'] += 1

            # Подсчет ошибок
            if log_entry['status'].startswith('4') or log_entry['status'].startswith('5'):
                ip_stats[ip]['errors'] += 1

            # Обнаружение угроз
            threats = self.detect_threats(log_entry)
            if threats:
                ip_stats[ip]['threats'] += len(threats)
                suspicious_servers[ip].append({
                    'line': line_num,
                    'threats': threats,
                    'request': log_entry['request'],
                    'timestamp': log_entry['timestamp']
                })

    return suspicious_servers, ip_stats


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