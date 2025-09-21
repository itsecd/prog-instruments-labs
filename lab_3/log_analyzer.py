import re
import csv
from collections import defaultdict
from datetime import datetime
from urllib.parse import unquote


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

    def parse_log_line(self, line):
        """Парсинг строки лога с помощью регулярного выражения"""
        match = self.log_pattern.match(line)
        if not match:
            return None

        return {
            'ip': match.group(1),
            'timestamp': match.group(2),
            'request': match.group(3),
            'status': match.group(4),
            'size': match.group(5),
            'referer': match.group(6),
            'user_agent': match.group(7)
        }

    def detect_threats(self, log_entry):
        """Обнаружение угроз в записи лога"""
        threats = []

        # Проверка всех паттернов
        for threat_type, patterns in self.patterns.items():
            for pattern in patterns:
                # Проверяем URL и User-Agent
                if re.search(pattern, log_entry['request'], re.IGNORECASE) or \
                        re.search(pattern, log_entry['user_agent'], re.IGNORECASE) or \
                        re.search(pattern, log_entry['referer'], re.IGNORECASE):
                    threats.append(threat_type)
                    break  # Не проверяем остальные паттерны для этого типа угрозы

        return threats



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

    def generate_report(self, suspicious_servers, ip_stats, output_file):
        """Генерация отчета в CSV"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['IP', 'Total Requests', 'Error Rate', 'Threat Count',
                          'Threat Types', 'First Detection', 'Last Detection']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for ip, entries in suspicious_servers.items():
                threat_types = set()
                for entry in entries:
                    threat_types.update(entry['threats'])

                timestamps = [datetime.strptime(entry['timestamp'].split()[0], '%d/%b/%Y:%H:%M:%S')
                              for entry in entries]

                writer.writerow({
                    'IP': ip,
                    'Total Requests': ip_stats[ip]['requests'],
                    'Error Rate': f"{(ip_stats[ip]['errors'] / ip_stats[ip]['requests']) * 100:.1f}%",
                    'Threat Count': ip_stats[ip]['threats'],
                    'Threat Types': ', '.join(threat_types),
                    'First Detection': min(timestamps).strftime('%Y-%m-%d %H:%M:%S'),
                    'Last Detection': max(timestamps).strftime('%Y-%m-%d %H:%M:%S')
                })



    def save_detailed_threats(self, suspicious_servers, output_file):
        """Сохранение детальной информации об угрозах"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['IP', 'Timestamp', 'Threat Type', 'Request', 'User Agent']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for ip, entries in suspicious_servers.items():
                for entry in entries:
                    for threat in entry['threats']:
                        writer.writerow({
                            'IP': ip,
                            'Timestamp': entry['timestamp'],
                            'Threat Type': threat,
                            'Request': entry['request'][:100],  # Ограничение длины
                            'User Agent': entry.get('user_agent', '')[:100]  # Ограничение длины
                        })


def main():
    log_file = 'access.log'
    results = analyze_log(log_file)

    for threat, entries in results.items():
        print(f"Обнаружена угроза: {threat}")
        for line_num, line in entries:
            print(f"Строка {line_num}: {line}")
        print("-" * 50)


if __name__ == '__main__':
    main()