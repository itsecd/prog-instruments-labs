"""Модуль с логикой парсинга логов."""

import csv
import json
from collections import Counter

from patterns import LOG_PATTERN


def parse_logs(input_file: str, valid_csv: str, broken_txt: str, stats_json: str) -> dict:
    """
    Парсит лог-файл, разделяет на валидные и битые, собирает статистику.

    Args:
        input_file: Путь к исходному файлу с логами.
        valid_csv: Путь для сохранения CSV-файла с валидными строками.
        broken_txt: Путь для сохранения файла с битыми строками.
        stats_json: Путь для сохранения файла со статистикой в JSON.

    Returns:
        Словарь со статистикой обработки.
    """
    
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
                
            match = LOG_PATTERN.match(line)
            
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
        
    return stats