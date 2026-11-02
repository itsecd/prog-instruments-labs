"""Тесты для парсера логов."""

import os
import tempfile
import unittest

from patterns import LOG_PATTERN
from parser import parse_logs


class TestLogPattern(unittest.TestCase):
    """Тесты регулярного выражения."""

    def test_valid_line_parses_correctly(self):
        """Валидная строка должна успешно парситься и извлекать все поля."""
        line = '192.168.1.10 - - [28/Aug/2025:14:32:10 +0800] "GET /index.html HTTP/1.1" 200 1024'
        match = LOG_PATTERN.match(line)
        
        self.assertIsNotNone(match)
        data = match.groupdict()
        
        self.assertEqual(data['ip'], '192.168.1.10')
        self.assertEqual(data['identd'], '-')
        self.assertEqual(data['userid'], '-')
        self.assertEqual(data['date'], '28/Aug/2025:14:32:10 +0800')
        self.assertEqual(data['method'], 'GET')
        self.assertEqual(data['url'], '/index.html')
        self.assertEqual(data['status'], '200')
        self.assertEqual(data['size'], '1024')

    def test_broken_line_does_not_match(self):
        """Битая строка не должна парситься."""
        line = 'BROKEN_LINE_1'
        match = LOG_PATTERN.match(line)
        self.assertIsNone(match)

    def test_missing_date_brackets_does_not_match(self):
        """Строка без квадратных скобок в дате не должна парситься."""
        line = '192.168.1.10 - - 28/Aug/2025:14:32:10 +0800 "GET /no-brackets HTTP/1.1" 200 100'
        match = LOG_PATTERN.match(line)
        self.assertIsNone(match)

    def test_invalid_http_method_does_not_match(self):
        """Строка с невалидным HTTP-методом (цифры) не должна парситься."""
        line = '192.168.1.10 - - [28/Aug/2025:14:32:10 +0800] "123 /index.html HTTP/1.1" 200 1024'
        match = LOG_PATTERN.match(line)
        self.assertIsNone(match)


class TestParseLogs(unittest.TestCase):
    """Тесты функции parse_logs."""

    def setUp(self):
        """Создает временную папку для тестов."""
        self.tmpdir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.tmpdir, 'input.txt')
        self.valid_csv = os.path.join(self.tmpdir, 'valid.csv')
        self.broken_txt = os.path.join(self.tmpdir, 'broken.txt')
        self.stats_json = os.path.join(self.tmpdir, 'stats.json')

    def test_parse_logs_returns_correct_stats(self):
        """Проверяет корректность собранной статистики."""
        with open(self.input_file, 'w', encoding='utf-8') as f:
            f.write('192.168.1.10 - - [28/Aug/2025:14:32:10 +0800] "GET /index.html HTTP/1.1" 200 1024\n')
            f.write('BROKEN_LINE_1\n')
            f.write('10.0.0.5 - - [28/Aug/2025:14:32:12 +0800] "POST /api/login HTTP/1.1" 200 512\n')

        stats = parse_logs(self.input_file, self.valid_csv, self.broken_txt, self.stats_json)

        self.assertEqual(stats['valid_lines'], 2)
        self.assertEqual(stats['broken_lines'], 1)
        self.assertEqual(stats['total_lines_processed'], 3)
        self.assertEqual(stats['status_codes']['200'], 2)

    def test_empty_lines_are_ignored(self):
        """Пустые строки не должны считаться ни валидными, ни битыми."""
        with open(self.input_file, 'w', encoding='utf-8') as f:
            f.write('192.168.1.10 - - [28/Aug/2025:14:32:10 +0800] "GET /index.html HTTP/1.1" 200 1024\n')
            f.write('\n')
            f.write('   \n')

        stats = parse_logs(self.input_file, self.valid_csv, self.broken_txt, self.stats_json)

        self.assertEqual(stats['valid_lines'], 1)
        self.assertEqual(stats['broken_lines'], 0)
        self.assertEqual(stats['total_lines_processed'], 1)

    def test_output_files_are_created(self):
        """Проверяет, что все выходные файлы создаются."""
        with open(self.input_file, 'w', encoding='utf-8') as f:
            f.write('192.168.1.10 - - [28/Aug/2025:14:32:10 +0800] "GET /index.html HTTP/1.1" 200 1024\n')

        parse_logs(self.input_file, self.valid_csv, self.broken_txt, self.stats_json)

        self.assertTrue(os.path.exists(self.valid_csv))
        self.assertTrue(os.path.exists(self.broken_txt))
        self.assertTrue(os.path.exists(self.stats_json))


if __name__ == '__main__':
    unittest.main()