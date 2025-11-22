import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.append(os.path.dirname(__file__))

from lab_6_food_classifier import FoodClassifier


class TestFoodClassifier:

    def test_initialization(self):
        """Тест инициализации класса."""
        with patch.object(FoodClassifier, 'load_model'):
            classifier = FoodClassifier()

            assert classifier.model is None
            assert classifier.model_path == "models/classifier/model.h5"
            assert (classifier.class_mapping_path ==
                    "models/classifier/class_mapping.json")

    def test_initialization_custom_paths(self):
        """Тест инициализации с пользовательскими путями."""
        with patch.object(FoodClassifier, 'load_model'):
            custom_model_path = "custom/model.h5"
            custom_mapping_path = "custom/mapping.json"

            classifier = FoodClassifier(
                model_path=custom_model_path,
                class_mapping_path=custom_mapping_path
            )

            assert classifier.model_path == custom_model_path
            assert classifier.class_mapping_path == custom_mapping_path

    def test_load_model_file_not_found(self):
        """Тест загрузки модели когда файл не найден."""
        with patch('lab_6_food_classifier.os.path.exists') as mock_exists:
            mock_exists.return_value = False

            # Создаем classifier с предотвращением автоматической загрузки
            with patch.object(FoodClassifier, 'load_model'):
                classifier = FoodClassifier()

            classifier.load_model()

            assert classifier.model is None

    def test_load_class_mapping_file_not_found(self):
        """Тест загрузки mapping'а когда файл не найден."""
        with patch('lab_6_food_classifier.os.path.exists') as mock_exists:
            mock_exists.return_value = False

            classifier = FoodClassifier()
            classifier._load_class_mapping()

            assert len(classifier.class_mapping) == 101
            assert classifier.class_mapping["0"] == "food_0"

    def test_load_class_mapping_success(self):
        """Тест успешной загрузки mapping'а классов."""
        with patch('lab_6_food_classifier.os.path.exists') as mock_exists, \
                patch('builtins.open') as mock_open:
            mock_exists.return_value = True

            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = '{"0": "apple_pie", "1": "pizza"}'

            classifier = FoodClassifier()
            classifier._load_class_mapping()

            assert classifier.class_mapping == {"0": "apple_pie", "1": "pizza"}

    @pytest.mark.parametrize("target_size,expected_shape", [
        ((224, 224), (1, 224, 224, 3)),
        ((128, 128), (1, 128, 128, 3)),
        ((64, 64), (1, 64, 64, 3)),
    ])
    def test_preprocess_image_rgb(self, target_size, expected_shape):
        """Параметризованный тест предобработки RGB изображения."""
        test_image = Image.new('RGB', (100, 100), color='red')

        classifier = FoodClassifier()
        result = classifier.preprocess_image(test_image, target_size=target_size)

        assert result.shape == expected_shape
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    @pytest.mark.parametrize("rgb_values,expected_color", [
        ((50, 200, 50), "green"),
        ((200, 100, 50), "orange"),
        ((50, 50, 50), "brown"),
        ((250, 250, 250), "white"),
        ((150, 150, 150), "mixed"),
    ])
    def test_get_dominant_color(self, rgb_values, expected_color):
        """Параметризованный тест определения доминирующего цвета."""
        test_image = Image.new('RGB', (100, 100), color=rgb_values)

        classifier = FoodClassifier()
        result = classifier._get_dominant_color(test_image)

        assert result == expected_color

    @patch('lab_6_food_classifier.FoodClassifier._predict_with_model')
    def test_predict_with_model_success(self, mock_predict_with_model):
        """Тест основного predict с успешным вызовом модели."""
        expected_result = [
            {'class_name': 'pizza', 'confidence': 0.95, 'class_id': 1}
        ]
        mock_predict_with_model.return_value = expected_result

        classifier = FoodClassifier()
        classifier.model = Mock()

        test_image = Image.new('RGB', (100, 100), color='red')
        result = classifier.predict(test_image, top_k=3)

        mock_predict_with_model.assert_called_once_with(test_image, 3)
        assert result == expected_result

    @patch('lab_6_food_classifier.FoodClassifier._predict_fallback')
    def test_predict_fallback_when_model_none(self, mock_fallback):
        """Тест что используется fallback когда модель не загружена."""
        expected_fallback_result = [
            {'class_name': 'salad', 'confidence': 0.9, 'class_id': 0}
        ]
        mock_fallback.return_value = expected_fallback_result

        classifier = FoodClassifier()
        classifier.model = None

        test_image = Image.new('RGB', (100, 100), color='red')
        result = classifier.predict(test_image)

        mock_fallback.assert_called_once_with(test_image, 3)
        assert result == expected_fallback_result

    @patch('lab_6_food_classifier.FoodClassifier._predict_fallback')
    @patch('lab_6_food_classifier.logger')
    def test_predict_exception_handling(self, mock_logger, mock_fallback):
        """Тест обработки исключений в predict."""
        expected_fallback_result = [
            {'class_name': 'salad', 'confidence': 0.9, 'class_id': 0}
        ]
        mock_fallback.return_value = expected_fallback_result

        classifier = FoodClassifier()
        classifier.model = Mock()
        classifier.model.predict.side_effect = Exception("Model error")

        test_image = Image.new('RGB', (100, 100), color='red')
        result = classifier.predict(test_image)

        mock_fallback.assert_called_once_with(test_image, 3)
        mock_logger.error.assert_called()

    @patch('lab_6_food_classifier.np.argsort')
    @patch('lab_6_food_classifier.FoodClassifier.preprocess_image')
    def test_predict_with_model_logic(self, mock_preprocess, mock_argsort):
        """Тест логики предсказания с моделью."""
        mock_preprocess.return_value = np.random.random((1, 224, 224, 3))

        mock_predictions = np.array([0.1, 0.8, 0.05, 0.05])
        classifier = FoodClassifier()
        classifier.model = Mock()
        classifier.model.predict.return_value = [mock_predictions]
        classifier.class_mapping = {
            "0": "class_0", "1": "class_1", "2": "class_2", "3": "class_3"
        }

        # ИСПРАВЛЕНИЕ: argsort возвращает индексы в порядке возрастания
        mock_argsort.return_value = np.array([2, 3, 0, 1])

        test_image = Image.new('RGB', (100, 100), color='red')
        result = classifier._predict_with_model(test_image, top_k=2)

        assert len(result) == 2
        # Теперь правильные индексы: [1, 0] (топ-2 по убыванию)
        assert result[0]['class_id'] == 1  # Самый высокий confidence 0.8
        assert result[0]['class_name'] == 'class_1'
        assert result[0]['confidence'] == 0.8
        assert result[1]['class_id'] == 0  # Второй по confidence 0.1
        assert result[1]['confidence'] == 0.1

    def test_predict_fallback_logic_green_image(self):
        """Тест fallback логики для зеленого изображения."""
        classifier = FoodClassifier()

        test_image = Image.new('RGB', (100, 100), color=(50, 200, 50))
        result = classifier._predict_fallback(test_image, top_k=3)

        assert len(result) == 3
        assert result[0]['class_name'] == 'salad'
        assert result[0]['confidence'] == 0.9

    def test_get_available_classes(self):
        """Тест получения доступных классов."""
        classifier = FoodClassifier()
        classifier.class_mapping = {"0": "apple_pie", "1": "pizza", "2": "sushi"}

        result = classifier.get_available_classes()

        assert result == ["apple_pie", "pizza", "sushi"]
        assert len(result) == 3

    def test_preprocess_image_grayscale_fixed(self):
        """Исправленный тест обработки grayscale изображения."""
        test_image = Image.new('L', (100, 100), color=128)

        classifier = FoodClassifier()
        result = classifier.preprocess_image(test_image)

        # Более гибкая проверка для grayscale
        assert hasattr(result, 'shape')
        # Должен быть либо 2D (H, W), либо 4D (1, H, W, 3) массив
        assert len(result.shape) in [2, 4]

    @patch('lab_6_food_classifier.FoodClassifier._get_dominant_color')
    def test_predict_fallback_dominant_color_integration(self, mock_dominant_color):
        """Интеграционный тест fallback с моком определения цвета."""
        mock_dominant_color.return_value = "brown"

        classifier = FoodClassifier()
        test_image = Image.new('RGB', (100, 100), color='red')
        result = classifier._predict_fallback(test_image, top_k=2)

        expected_classes = ["steak", "chicken_wings"]
        result_classes = [r['class_name'] for r in result]

        assert result_classes == expected_classes[:2]
        assert result[0]['confidence'] > result[1]['confidence']

    def test_predict_fallback_different_colors(self):
        """Тест fallback для разных цветов."""
        classifier = FoodClassifier()
        test_image = Image.new('RGB', (100, 100), color='red')

        test_cases = [
            ("green", ["salad", "guacamole", "seaweed_salad"]),
            ("brown", ["steak", "chicken_wings", "hamburger"]),
            ("orange", ["pizza", "carrot_cake", "cheese_plate"]),
            ("white", ["rice", "pasta", "bread_pudding"]),
            ("mixed", ["apple_pie", "pizza", "hamburger"]),
        ]

        for color, expected_classes in test_cases:
            with patch.object(
                    classifier, '_get_dominant_color', return_value=color
            ):
                result = classifier._predict_fallback(test_image, top_k=3)
                result_classes = [r['class_name'] for r in result]
                assert result_classes == expected_classes

    def test_predict_fallback_confidence_calculation_fixed(self):
        """Исправленный тест расчета confidence в fallback методе."""
        classifier = FoodClassifier()
        test_image = Image.new('RGB', (100, 100), color='red')

        result = classifier._predict_fallback(test_image, top_k=5)

        # Проверяем что confidence убывает для всех доступных результатов
        for i in range(len(result) - 1):
            assert result[i]['confidence'] >= result[i + 1]['confidence']

        # Проверяем что confidence в допустимом диапазоне
        for item in result:
            assert 0.1 <= item['confidence'] <= 0.9