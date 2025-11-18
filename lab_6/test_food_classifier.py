import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import json
import os
import tensorflow as tf
from your_module import FoodClassifier  # замените на ваш импорт


class TestFoodClassifier:

    def test_initialization(self):
        """Тест инициализации класса"""
        classifier = FoodClassifier()

        assert classifier.model is None
        assert classifier.class_mapping == {}
        assert classifier.model_path == "models/classifier/model.h5"
        assert classifier.class_mapping_path == "models/classifier/class_mapping.json"

    def test_initialization_custom_paths(self):
        """Тест инициализации с пользовательскими путями"""
        custom_model_path = "custom/model.h5"
        custom_mapping_path = "custom/mapping.json"

        classifier = FoodClassifier(
            model_path=custom_model_path,
            class_mapping_path=custom_mapping_path
        )

        assert classifier.model_path == custom_model_path
        assert classifier.class_mapping_path == custom_mapping_path

    @patch('your_module.tf.keras.models.load_model')
    @patch('builtins.open')
    @patch('your_module.os.path.exists')
    def test_load_model_success(self, mock_exists, mock_open, mock_load_model):
        """Тест успешной загрузки модели и mapping'а"""
        # Мокаем существование файлов
        mock_exists.return_value = True

        # Мокаем загрузку модели
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Мокаем загрузку class mapping
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_file.read.return_value = '{"0": "apple_pie", "1": "pizza"}'

        classifier = FoodClassifier()
        classifier.load_model()

        # Проверяем вызовы
        mock_load_model.assert_called_once_with("models/classifier/model.h5", compile=False)
        mock_open.assert_called_once_with("models/classifier/class_mapping.json", 'r', encoding='utf-8')
        assert classifier.model == mock_model
        assert classifier.class_mapping == {"0": "apple_pie", "1": "pizza"}

    @patch('your_module.logger')
    @patch('your_module.os.path.exists')
    def test_load_model_not_found(self, mock_exists, mock_logger):
        """Тест загрузки модели когда файл не найден"""
        mock_exists.return_value = False

        classifier = FoodClassifier()
        classifier.load_model()

        # Проверяем что модель осталась None
        assert classifier.model is None
        # Проверяем что было логирование предупреждения
        mock_logger.warning.assert_called()

    @patch('builtins.open')
    @patch('your_module.os.path.exists')
    def test_load_class_mapping_file_not_found(self, mock_exists, mock_open):
        """Тест загрузки mapping'а когда файл не найден"""
        mock_exists.return_value = False

        classifier = FoodClassifier()
        classifier._load_class_mapping()

        # Проверяем что создан mapping по умолчанию
        assert len(classifier.class_mapping) == 101
        assert classifier.class_mapping["0"] == "food_0"
        assert classifier.class_mapping["100"] == "food_100"

    @patch('builtins.open')
    @patch('your_module.os.path.exists')
    def test_load_class_mapping_json_error(self, mock_exists, mock_open):
        """Тест обработки ошибки JSON при загрузке mapping'а"""
        mock_exists.return_value = True
        mock_open.side_effect = Exception("JSON decode error")

        classifier = FoodClassifier()
        classifier._load_class_mapping()

        # Проверяем fallback mapping
        assert len(classifier.class_mapping) == 101

    @pytest.mark.parametrize("target_size,expected_shape", [
        ((224, 224), (1, 224, 224, 3)),
        ((128, 128), (1, 128, 128, 3)),
        ((64, 64), (1, 64, 64, 3)),
    ])
    def test_preprocess_image(self, target_size, expected_shape):
        """Параметризованный тест предобработки изображения"""
        # Создаем тестовое изображение
        test_image = Image.new('RGB', (100, 100), color='red')

        classifier = FoodClassifier()
        result = classifier.preprocess_image(test_image, target_size=target_size)

        # Проверяем форму и нормализацию
        assert result.shape == expected_shape
        assert result.max() <= 1.0  # Проверяем нормализацию
        assert result.min() >= 0.0

    @pytest.mark.parametrize("rgb_values,expected_color", [
        ((50, 200, 50), "green"),  # Высокий зеленый
        ((200, 100, 50), "orange"),  # Высокий красный
        ((50, 50, 50), "brown"),  # Темные цвета
        ((250, 250, 250), "white"),  # Светлые цвета
        ((150, 150, 150), "mixed"),  # Смешанные цвета
    ])
    def test_get_dominant_color(self, rgb_values, expected_color):
        """Параметризованный тест определения доминирующего цвета"""
        # Создаем изображение с заданным цветом
        test_image = Image.new('RGB', (100, 100), color=rgb_values)

        classifier = FoodClassifier()
        result = classifier._get_dominant_color(test_image)

        assert result == expected_color

    @patch('your_module.FoodClassifier._predict_with_model')
    def test_predict_with_model_success(self, mock_predict_with_model):
        """Тест основного predict с успешным вызовом модели"""
        # Мокаем успешный результат
        expected_result = [{'class_name': 'pizza', 'confidence': 0.95, 'class_id': 1}]
        mock_predict_with_model.return_value = expected_result

        classifier = FoodClassifier()
        classifier.model = Mock()  # Мокаем что модель загружена

        test_image = Image.new('RGB', (100, 100), color='red')
        result = classifier.predict(test_image, top_k=3)

        mock_predict_with_model.assert_called_once_with(test_image, 3)
        assert result == expected_result

    @patch('your_module.FoodClassifier._predict_fallback')
    @patch('your_module.logger')
    def test_predict_fallback_when_model_none(self, mock_logger, mock_fallback):
        """Тест что используется fallback когда модель не загружена"""
        expected_fallback_result = [{'class_name': 'salad', 'confidence': 0.9, 'class_id': 0}]
        mock_fallback.return_value = expected_fallback_result

        classifier = FoodClassifier()
        classifier.model = None  # Модель не загружена

        test_image = Image.new('RGB', (100, 100), color='red')
        result = classifier.predict(test_image)

        mock_fallback.assert_called_once_with(test_image, 3)
        assert result == expected_fallback_result

    @patch('your_module.FoodClassifier._predict_fallback')
    @patch('your_module.logger')
    def test_predict_exception_handling(self, mock_logger, mock_fallback):
        """Тест обработки исключений в predict"""
        expected_fallback_result = [{'class_name': 'salad', 'confidence': 0.9, 'class_id': 0}]
        mock_fallback.return_value = expected_fallback_result

        classifier = FoodClassifier()
        classifier.model = Mock()
        # Мокаем исключение при вызове модели
        classifier.model.predict.side_effect = Exception("Model error")

        test_image = Image.new('RGB', (100, 100), color='red')
        result = classifier.predict(test_image)

        # Проверяем что был вызван fallback
        mock_fallback.assert_called_once_with(test_image, 3)
        # Проверяем логирование ошибки
        mock_logger.error.assert_called()

    @patch('your_module.np.argsort')
    @patch('your_module.FoodClassifier.preprocess_image')
    def test_predict_with_model_logic(self, mock_preprocess, mock_argsort):
        """Тест логики предсказания с моделью"""
        # Мокаем предобработку изображения
        mock_preprocess.return_value = np.random.random((1, 224, 224, 3))

        # Мокаем предсказания модели
        mock_predictions = np.array([0.1, 0.8, 0.05, 0.05])  # 4 класса
        classifier = FoodClassifier()
        classifier.model = Mock()
        classifier.model.predict.return_value = [mock_predictions]
        classifier.class_mapping = {"0": "class_0", "1": "class_1", "2": "class_2", "3": "class_3"}

        # Мокаем argsort для top_k=2
        mock_argsort.return_value = np.array([1, 0])  # Индексы отсортированные по убыванию confidence

        test_image = Image.new('RGB', (100, 100), color='red')
        result = classifier._predict_with_model(test_image, top_k=2)

        # Проверяем структуру результата
        assert len(result) == 2
        assert result[0]['class_id'] == 1
        assert result[0]['class_name'] == 'class_1'
        assert result[0]['confidence'] == 0.8

    def test_predict_fallback_logic_green_image(self):
        """Тест fallback логики для зеленого изображения"""
        classifier = FoodClassifier()

        # Создаем зеленое изображение
        test_image = Image.new('RGB', (100, 100), color=(50, 200, 50))

        result = classifier._predict_fallback(test_image, top_k=3)

        assert len(result) == 3
        # Для зеленого изображения ожидаем салат и т.д.
        assert result[0]['class_name'] == 'salad'
        assert result[0]['confidence'] == 0.9

    def test_get_available_classes(self):
        """Тест получения доступных классов"""
        classifier = FoodClassifier()
        classifier.class_mapping = {"0": "apple_pie", "1": "pizza", "2": "sushi"}

        result = classifier.get_available_classes()

        assert result == ["apple_pie", "pizza", "sushi"]
        assert len(result) == 3

    def test_preprocess_image_grayscale_conversion(self):
        """Тест обработки grayscale изображения"""
        # Создаем grayscale изображение
        test_image = Image.new('L', (100, 100), color=128)

        classifier = FoodClassifier()
        result = classifier.preprocess_image(test_image)

        # Проверяем что изображение конвертировано в RGB и имеет правильную форму
        assert result.shape == (1, 224, 224, 3)

    @patch('your_module.FoodClassifier._get_dominant_color')
    def test_predict_fallback_dominant_color_integration(self, mock_dominant_color):
        """Интеграционный тест fallback с моком определения цвета"""
        mock_dominant_color.return_value = "brown"

        classifier = FoodClassifier()
        test_image = Image.new('RGB', (100, 100), color='red')
        result = classifier._predict_fallback(test_image, top_k=2)

        # Для brown цвета ожидаем steak, chicken_wings, hamburger
        expected_classes = ["steak", "chicken_wings"]
        result_classes = [r['class_name'] for r in result]

        assert result_classes == expected_classes[:2]
        # Проверяем что confidence убывает
        assert result[0]['confidence'] > result[1]['confidence']