import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Добавляем путь для импорта модуля
sys.path.append('.')
from lab_6 import AnalysisConfig, totalServed, negativeCount, negativeShare
from lab_6 import _filter_negative_themes, _extract_negative_theme_details, _get_negative_themes_with_details


class TestAnalysisConfig:
    """Тесты для класса конфигурации"""

    def test_config_constants(self):
        """Проверка основных констант конфигурации"""
        assert AnalysisConfig.NEGATIVE_THEME_PATTERN == "Недовольство"
        assert AnalysisConfig.CSI_BASE_VALUE == 87.97
        assert AnalysisConfig.CSI_MIN_VALUE == 0
        assert AnalysisConfig.CSI_MAX_VALUE == 100


class TestFilterFunctions:
    """Тесты для функций фильтрации"""

    @pytest.fixture
    def sample_data(self):
        """Фикстура с тестовыми данными"""
        return pd.DataFrame({
            'Тема обращения': [
                'Недовольство/Качество связи',
                'Недовольство/Тарифы',
                'Техническая поддержка',
                'Недовольство/Обслуживание',
                'Консультация'
            ],
            'Дата обращения': [
                '2024-01-01', '2024-01-02', '2024-01-03',
                '2024-01-04', '2024-01-05'
            ],
            'SA': [4.5, 3.8, 4.2, 3.5, 4.7],
            'CES': [4.0, 3.2, 4.5, 3.0, 4.8],
            'NPS': [8, 6, 9, 5, 10],
            'ARPU': ['B2C Low', 'B2C Mid', 'VIP', 'VIP adv', 'Platinum'],
            'Регион': ['Москва', 'СПб', 'Самара', 'Новосибирск', 'Саратов'],
            'Район': ['Октябрьский', 'Кировский', 'Западный', 'Центральный', 'Северный']
        })

    def test_filter_negative_themes(self, sample_data):
        """Тест фильтрации негативных обращений"""
        result = _filter_negative_themes(sample_data)
        assert len(result) == 3
        assert all('Недовольство' in theme for theme in result['Тема обращения'])

    def test_extract_negative_theme_details(self, sample_data):
        """Тест извлечения деталей тем"""
        negative_df = _filter_negative_themes(sample_data)
        result = _extract_negative_theme_details(negative_df)

        assert 'Тема' in result.columns
        assert result['Тема'].isna().sum() == 0
        assert 'Качество связи' in result['Тема'].values

    def test_get_negative_themes_with_details(self, sample_data):
        """Тест получения негативных обращений с деталями"""
        result = _get_negative_themes_with_details(sample_data)
        assert len(result) == 3
        assert 'Тема' in result.columns


class TestBasicMetrics:
    """Тесты для базовых метрик"""

    @pytest.fixture
    def metrics_data(self):
        """Фикстура для тестирования метрик"""
        return pd.DataFrame({
            'Тема обращения': [
                'Недовольство/Качество', 'Недовольство/Тарифы',
                'Поддержка', 'Консультация'
            ],
            'SA': [4.0, 3.5, 4.5, 4.8],
            'CES': [3.5, 3.0, 4.2, 4.7],
            'NPS': [7, 6, 9, 10]
        })

    def test_total_served(self, metrics_data):
        """Тест подсчета общего количества обслуженных клиентов"""
        result = totalServed(metrics_data)
        assert result == 4

    def test_total_served_empty(self):
        """Тест подсчета для пустого DataFrame"""
        result = totalServed(pd.DataFrame())
        assert result == 0

    def test_negative_count(self, metrics_data):
        """Тест подсчета негативных обращений"""
        result = negativeCount(metrics_data)
        assert result == 2

    def test_negative_share(self, metrics_data):
        """Тест расчета доли негативных обращений"""
        result = negativeShare(metrics_data)
        expected = (2 / 4) * 100
        assert result == expected

    def test_negative_share_zero_division(self):
        """Тест обработки деления на ноль"""
        empty_df = pd.DataFrame({'Тема обращения': []})
        result = negativeShare(empty_df)
        assert result == 0