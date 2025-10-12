import pytest
import json
import pandas as pd
from io import StringIO, BytesIO
import sys
import os

# Добавляем корневую директорию в путь Python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# АБСОЛЮТНЫЕ импорты
from app import (
    allowed_file,
    convert_csv_to_json,
    convert_csv_to_xml,
    convert_json_to_csv,
    convert_json_to_xml,
    convert_to_txt,
)


class TestAllowedFile:
    """Тесты функции проверки разрешенных файлов"""

    def test_allowed_file_valid_extensions(self):
        """Тест разрешенных расширений"""
        assert allowed_file("test.csv") == True
        assert allowed_file("data.json") == True
        assert allowed_file("image.png") == True
        assert allowed_file("document.pdf") == True

    def test_allowed_file_invalid_extensions(self):
        """Тест запрещенных расширений"""
        assert allowed_file("script.exe") == False
        assert allowed_file("malicious.php") == False
        assert allowed_file("file.bat") == False
        assert allowed_file("no_extension") == False


class TestCSVConversions:
    """Тесты конвертации CSV"""

    def test_convert_csv_to_json_basic(self, sample_csv_content):
        """Тест базовой конвертации CSV в JSON"""
        result = convert_csv_to_json(sample_csv_content)

        # Проверяем что результат валидный JSON
        parsed_json = json.loads(result)
        assert isinstance(parsed_json, list)
        assert len(parsed_json) == 2
        assert parsed_json[0]["name"] == "John"
        # pandas может автоматически определить тип числа, поэтому проверяем значение, а не тип
        assert parsed_json[0]["age"] == 30
        assert parsed_json[0]["city"] == "New York"
        assert parsed_json[1]["name"] == "Jane"
        assert parsed_json[1]["age"] == 25
        assert parsed_json[1]["city"] == "London"

    def test_convert_csv_to_xml_structure(self, sample_csv_content):
        """Тест структуры XML при конвертации из CSV"""
        result = convert_csv_to_xml(sample_csv_content)

        assert "<?xml version=" in result
        assert "<root>" in result
        assert "<record>" in result
        assert "<name>John</name>" in result
        assert "<age>30</age>" in result
        assert "<city>New York</city>" in result


class TestJSONConversions:
    """Тесты конвертации JSON"""

    def test_convert_json_to_csv_basic(self, sample_json_content):
        """Тест конвертации JSON в CSV"""
        result = convert_json_to_csv(sample_json_content)

        # Проверяем CSV структуру
        lines = result.strip().split('\n')
        assert len(lines) == 3  # header + 2 data rows
        assert "name,age" in lines[0]
        assert "John,30" in lines[1]
        assert "Jane,25" in lines[2]

    def test_convert_json_to_xml_structure(self, sample_json_content):
        """Тест конвертации JSON в XML"""
        result = convert_json_to_xml(sample_json_content)

        assert "<root>" in result
        assert "<name>John</name>" in result
        assert "<age>30</age>" in result


class TestTextConversions:
    """Тесты конвертации в текст"""

    def test_convert_csv_to_txt(self, sample_csv_content):
        """Тест конвертации CSV в текст"""
        result = convert_to_txt(sample_csv_content, "csv")

        assert "name" in result
        assert "John" in result
        assert "30" in result

    def test_convert_json_to_txt(self, sample_json_content):
        """Тест конвертации JSON в текст"""
        result = convert_to_txt(sample_json_content, "json")

        # Проверяем что результат содержит данные JSON
        assert "John" in result
        assert "30" in result