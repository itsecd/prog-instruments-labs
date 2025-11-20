#!/usr/bin/env python3
"""
Basic unit tests for nmap_gui_scan.py
"""

import sys
import os
import pytest

# Добавляем родительскую директорию в путь для импорта основного модуля
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nmap_gui_scan import find_nmap, sanitize_arg


class TestBasicFunctions:
    """Тесты базовых функций"""

    def test_find_nmap_exists(self):
        """Тест поиска nmap в системе"""
        result = find_nmap()
        # nmap может быть или не быть в системе, но функция должна возвращать строку или None
        assert result is None or isinstance(result, str)

    def test_sanitize_arg_normal_input(self):
        """Тест очистки нормальных аргументов"""
        test_cases = [
            ("192.168.1.1", "192.168.1.1"),
            ("  example.com  ", "example.com"),
            ("", ""),
            ("   ", ""),
        ]

        for input_arg, expected in test_cases:
            result = sanitize_arg(input_arg)
            assert result == expected, f"Failed for input: '{input_arg}'"

    def test_sanitize_arg_special_chars(self):
        """Тест очистки аргументов со специальными символами"""
        # Функция sanitize_arg только удаляет пробелы, специальные символы не обрабатываются
        test_input = "example.com; rm -rf /"
        result = sanitize_arg(test_input)
        expected = "example.com; rm -rf /"
        assert result == expected

    def test_find_nmap_mocked_found(self, mocker):
        """Тест поиска nmap с моком (найден)"""
        # Мокаем shutil.which чтобы возвращать фиктивный путь
        mock_which = mocker.patch('nmap_gui_scan.shutil.which')
        mock_which.return_value = "/usr/bin/nmap"

        result = find_nmap()
        assert result == "/usr/bin/nmap"
        mock_which.assert_called_once_with("nmap")

    def test_find_nmap_mocked_not_found(self, mocker):
        """Тест поиска nmap с моком (не найден)"""
        mock_which = mocker.patch('nmap_gui_scan.shutil.which')
        mock_which.return_value = None

        result = find_nmap()
        assert result is None
        mock_which.assert_called_once_with("nmap")

    def test_sanitize_arg_with_none(self):
        """Тест очистки аргумента с None значением"""
        # Добавляем тест для граничного случая
        result = sanitize_arg(None)
        # Ожидаем, что функция обработает None без ошибки
        assert result is None