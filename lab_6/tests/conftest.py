import pytest
import sys
import os
from io import BytesIO
from unittest.mock import MagicMock

# Добавляем корневую директорию в путь Python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Теперь импортируем АБСОЛЮТНО
from app import app as flask_app


@pytest.fixture
def app():
    """Фикстура Flask приложения"""
    flask_app.config['TESTING'] = True
    flask_app.config['WTF_CSRF_ENABLED'] = False
    flask_app.config['UPLOAD_FOLDER'] = 'temp_uploads_test'

    # Создаем тестовую директорию для загрузок
    os.makedirs(flask_app.config['UPLOAD_FOLDER'], exist_ok=True)

    yield flask_app

    # Очистка после тестов
    import shutil
    if os.path.exists(flask_app.config['UPLOAD_FOLDER']):
        shutil.rmtree(flask_app.config['UPLOAD_FOLDER'])


@pytest.fixture
def client(app):
    """Фикстура тестового клиента"""
    return app.test_client()


@pytest.fixture
def sample_csv_content():
    """Фикстура с примером CSV контента"""
    return "name,age,city\nJohn,30,New York\nJane,25,London"


@pytest.fixture
def sample_json_content():
    """Фикстура с примером JSON контента"""
    return '[{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]'


@pytest.fixture
def sample_xml_content():
    """Фикстура с примером XML контента"""
    return '<?xml version="1.0"?><root><name>John</name><age>30</age></root>'


@pytest.fixture
def mock_pdf_file():
    """Фикстура мока PDF файла"""
    mock_file = MagicMock()
    mock_file.read.return_value = b"%PDF-1.4 fake PDF content"
    mock_file.seek.return_value = None
    return mock_file


@pytest.fixture
def mock_image_file():
    """Фикстура мока изображения"""
    mock_file = MagicMock()
    mock_file.read.return_value = b"fake image content"
    return mock_file