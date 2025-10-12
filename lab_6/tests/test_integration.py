import pytest
import json
from io import BytesIO
import tempfile
import os
import sys

# Добавляем корневую директорию в путь Python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# АБСОЛЮТНЫЕ импорты
from app import temp_files

class TestFlaskEndpoints:
    """Тесты Flask endpoints"""

    def test_index_route(self, client):
        """Тест главной страницы"""
        response = client.get('/')
        assert response.status_code == 200

    def test_convert_route_no_file(self, client):
        """Тест конвертации без файла"""
        response = client.post('/convert', data={})
        assert response.status_code == 400
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'No file uploaded' in json_data['error']

    def test_convert_route_invalid_file_type(self, client):
        """Тест конвертации с запрещенным типом файла"""
        test_data = {
            'file': (BytesIO(b'content'), 'test.exe'),
            'target_format': 'json'
        }

        response = client.post('/convert',
                           data=test_data,
                           content_type='multipart/form-data')

        assert response.status_code == 400
        json_data = response.get_json()
        assert 'File type not supported' in json_data['error']

    def test_convert_route_success_csv_to_json(self, client):
        """Тест успешной конвертации CSV в JSON"""
        csv_content = "name,age\nJohn,30\nJane,25"
        test_data = {
            'file': (BytesIO(csv_content.encode('utf-8')), 'test.csv'),
            'target_format': 'json'
        }

        response = client.post('/convert',
                           data=test_data,
                           content_type='multipart/form-data')

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['success'] == True
        assert 'file_id' in json_data
        assert 'filename' in json_data
        assert json_data['filename'] == 'test.json'

    def test_download_route_file_not_found(self, client):
        """Тест скачивания несуществующего файла"""
        response = client.get('/download/invalid_id/filename.json')
        assert response.status_code == 404
        json_data = response.get_json()
        assert 'error' in json_data

    def test_convert_route_missing_target_format(self, client):
        """Тест конвертации без указания целевого формата"""
        test_data = {
            'file': (BytesIO(b'content'), 'test.csv')
            # Нет target_format
        }

        response = client.post('/convert',
                           data=test_data,
                           content_type='multipart/form-data')

        assert response.status_code == 400
        json_data = response.get_json()
        assert 'Target format not specified' in json_data['error']