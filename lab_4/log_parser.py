import re
import csv
import json
import argparse
from collections import Counter


def create_reg():
    """Создает и компилирует регулярное выражение для парсинга логов."""
    ip = r"(?P<ip>\d{1,3}(?:\.\d{1,3}){3})"
    name = r"\s+(?P<identd>\S+)\s+(?P<userid>\S+)"
    date_time = r"\s+\[(?P<date>\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\s[+-]\d{4})\]"
    http = r'\s+"(?P<method>[A-Z]+)\s+(?P<url>[^\s"]+)\s+HTTP/\d\.\d"'
    status_and_size = r"\s+(?P<status>\d{3})\s+(?P<size>\d+)"
    
    return re.compile(f"^{ip}{name}{date_time}{http}{status_and_size}")


def parse_logs(input_file, valid_csv, broken_txt, stats_json):
    """Парсит лог-файл, разделяет на валидные и битые, собирает статистику."""
    
    log_pattern = create_reg()
    
    valid_count = 0
    broken_count = 0
    
    status_counter = Counter()
    
    print(f"Чтение файла: {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(valid_csv, 'w', newline='', encoding='utf-8') as f_valid, \
         open(broken_txt, 'w', encoding='utf-8') as f_broken:
        
        writer = csv.writer(f_valid)
        writer.writerow(['ip', 'identd', 'userid', 'date', 'method', 'url', 'status', 'size'])
        
        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue  
                
            match = log_pattern.match(line)
            
            if match:
                data = match.groupdict()
                
                writer.writerow([
                    data['ip'], data['identd'], data['userid'], 
                    data['date'], data['method'], data['url'], 
                    data['status'], data['size']
                ])
                
                status_counter[data['status']] += 1
                valid_count += 1
            else:
                f_broken.write(f"Строка {line_num}: {line}\n")
                broken_count += 1
                
    stats = {
        "total_lines_processed": valid_count + broken_count,
        "valid_lines": valid_count,
        "broken_lines": broken_count,
        "status_codes": dict(status_counter)
    }
    
    with open(stats_json, 'w', encoding='utf-8') as f_stats:
        json.dump(stats, f_stats, indent=4, ensure_ascii=False)
        
    print(f"Всего строк обработано: {stats['total_lines_processed']}")
    print(f"Валидных строк (сохранены в {valid_csv}): {valid_count}")
    print(f"Битых строк (сохранены в {broken_txt}): {broken_count}")
    print(f"Статистика по кодам ответа: {dict(status_counter)}")
    print(f"Полная статистика сохранена в {stats_json}")


def main():
    """Главная функция, разбирающая аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Парсер логов веб-сервера. Извлекает данные с помощью регулярных выражений и собирает статистику.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_file", 
        help="Путь к исходному файлу с логами (например, server.log)"
    )
    parser.add_argument(
        "-o", "--output", 
        default="parsed_logs.csv", 
        help="Путь для сохранения CSV-файла с валидными строками"
    )
    parser.add_argument(
        "-b", "--broken", 
        default="broken_logs.txt", 
        help="Путь для сохранения файла с битыми строками"
    )
    parser.add_argument(
        "-s", "--stats", 
        default="stats.json", 
        help="Путь для сохранения файла со статистикой (JSON)"
    )
    
    args = parser.parse_args()
    
    parse_logs(args.input_file, args.output, args.broken, args.stats)


if __name__ == "__main__":
    main()