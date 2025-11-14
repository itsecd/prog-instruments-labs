import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys

# Добавляем путь для импорта модуля
sys.path.append('.')
from lab_6 import AnalysisConfig, totalServed, negativeCount, negativeShare
from lab_6 import _filter_negative_themes, _extract_negative_theme_details, _get_negative_themes_with_details
from lab_6 import csiIndex
from lab_6 import arpuSegments, arpuNegativeThemes
from lab_6 import forecastNegativeThemes, forecastDeviation
from lab_6 import regionDistrictAnalysis, forecastNegativeThemes, arpuNegativeThemes
from lab_6 import totalServed, negativeCount, negativeShare, csiIndex, arpuSegments


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


    class TestCSICalculation:
        """Тесты для расчета CSI индекса"""

        @pytest.fixture
        def csi_data(self):
            """Фикстура для тестирования CSI"""
            return pd.DataFrame({
                'SA': [4.5, 4.0, 3.5, 4.8, 3.2],
                'CES': [4.2, 3.8, 3.0, 4.5, 2.8],
                'NPS': [9, 8, 6, 10, 5],
                'Тема обращения': ['Тест'] * 5
            })


        def test_csi_index_calculation(self, csi_data):
            """Тест расчета CSI индекса"""
            result = csiIndex(csi_data)

            # Проверяем что результат в допустимых пределах
            assert 0 <= result <= 100
            assert isinstance(result, float)


        def test_csi_index_empty_data(self):
            """Тест CSI для пустых данных"""
            empty_df = pd.DataFrame({'SA': [], 'CES': [], 'NPS': []})
            result = csiIndex(empty_df)
            assert result == 0

        @pytest.mark.parametrize("sa,ces,nps,expected_range", [
            (5.0, 5.0, 10, (87.0, 89.0)),  # Высокие оценки
            (1.0, 1.0, 0, (85.0, 87.0)),  # Низкие оценки
            (3.0, 3.0, 5, (86.0, 88.0)),  # Средние оценки
        ])


        def test_csi_parameterized(self, sa, ces, nps, expected_range):
            """Параметризованный тест CSI с различными входными данными"""
            test_data = pd.DataFrame({
                'SA': [sa], 'CES': [ces], 'NPS': [nps], 'Тема обращения': ['Test']
            })

            result = csiIndex(test_data)
            assert expected_range[0] <= result <= expected_range[1]


class TestARPUFunctions:
    """Тесты для функций работы с ARPU сегментами"""

    @pytest.fixture
    def arpu_data(self):
        """Фикстура с данными для тестирования ARPU"""
        return pd.DataFrame({
            'Тема обращения': [
                'Недовольство/Качество', 'Недовольство/Тарифы', 'Поддержка',
                'Недовольство/Обслуживание', 'Консультация', 'Недовольство/Качество'
            ],
            'ARPU': ['B2C Low', 'B2C Mid', 'B2C Low', 'VIP', 'VIP adv', 'Platinum']
        })


    def test_arpu_segments(self, arpu_data):
        """Тест подсчета обращений по ARPU сегментам"""
        result = arpuSegments(arpu_data)

        # Проверяем что все ожидаемые ключи присутствуют
        expected_keys = [
            'negative_b2c_low', 'total_b2c_low',
            'negative_b2c_mid', 'total_b2c_mid',
            'negative_vip', 'total_vip',
            'negative_vip_adv', 'total_vip_adv',
            'negative_platinum', 'total_platinum'
        ]

        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], int)


    def test_arpu_negative_themes_all_segments(self, arpu_data):
        """Тест построения графика тем для всех сегментов"""
        result = arpuNegativeThemes(arpu_data, segment="Все")

        assert 'plot' in result
        assert 'svg' in result['plot']


    @pytest.mark.parametrize("segment", ["B2C Low", "B2C Mid", "VIP", "VIP adv", "Platinum"])
    def test_arpu_negative_themes_by_segment(self, arpu_data, segment):
        """Параметризованный тест построения графиков по разным сегментам"""
        result = arpuNegativeThemes(arpu_data, segment=segment)

        assert 'plot' in result
        assert 'svg' in result['plot']


class TestForecastFunctions:
    """Тесты для функций прогнозирования"""

    @pytest.fixture
    def forecast_data(self):
        """Фикстура с данными для прогнозирования"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        return pd.DataFrame({
            'Тема обращения': [
                'Недовольство/Качество', 'Недовольство/Тарифы', 'Недовольство/Обслуживание',
                'Недовольство/Качество', 'Недовольство/Тарифы', 'Поддержка',
                'Недовольство/Обслуживание', 'Консультация', 'Недовольство/Качество', 'Поддержка'
            ],
            'Дата обращения': dates
        })


    def test_forecast_negative_themes_plot(self, forecast_data):
        """Тест функции прогнозирования с возвратом графика"""
        result = forecastNegativeThemes(forecast_data, return_type='plot')
        assert result is not None
        assert 'svg' in result


    def test_forecast_negative_themes_forecast(self, forecast_data):
        """Тест функции прогнозирования с возвратом прогноза"""
        result = forecastNegativeThemes(forecast_data, return_type='forecast')

        assert 'forecast_today' in result
        assert 'forecast_tomorrow' in result
        assert 'r_squared' in result
        assert isinstance(result['forecast_today'], float)
        assert isinstance(result['forecast_tomorrow'], float)


    def test_forecast_negative_themes_both(self, forecast_data):
        """Тест функции прогнозирования с возвратом обоих результатов"""
        result = forecastNegativeThemes(forecast_data, return_type='both')

        assert 'plot' in result
        assert 'forecast' in result
        assert 'svg' in result['plot']


    @patch('lab_2.linregress')
    def test_forecast_negative_themes_with_mock(self, mock_linregress, forecast_data):
        """Тест прогнозирования с моком linregress"""
        # Настраиваем мок
        mock_linregress.return_value = Mock(
            slope=0.5, intercept=1.0, rvalue=0.8, pvalue=0.05, stderr=0.1
        )

        result = forecastNegativeThemes(forecast_data, return_type='forecast')

        mock_linregress.assert_called_once()

        # Проверяем структуру результата
        assert 'forecast_today' in result
        assert 'forecast_tomorrow' in result
        assert 'r_squared' in result


class TestRegionAnalysis:
    """Тесты для анализа по регионам и районам"""

    @pytest.fixture
    def region_data(self):
        """Фикстура с региональными данными"""
        return pd.DataFrame({
            'Тема обращения': [
                'Недовольство/Качество', 'Недовольство/Тарифы', 'Недовольство/Обслуживание',
                'Недовольство/Качество', 'Недовольство/Тарифы', 'Недовольство/Обслуживание'
            ],
            'Регион': ['Москва', 'Москва', 'СПб', 'СПб', 'Новосибирск', 'Новосибирск'],
            'Район': ['Центр', 'Западный', 'Василеостровский', 'Петроградский', 'Центральный', 'Железнодорожный'],
            'ARPU': ['B2C Low'] * 6
        })


    def test_region_analysis_all(self, region_data):
        """Тест анализа всех регионов"""
        result = regionDistrictAnalysis(region_data)

        assert 'regions' in result
        assert 'themes' in result
        assert 'table' in result
        assert 'stats' in result
        assert len(result['regions']) > 0


    def test_region_analysis_with_region_filter(self, region_data):
        """Тест анализа с фильтром по региону"""
        result = regionDistrictAnalysis(region_data, region="Москва")

        assert result['current_region'] == "Москва"
        assert 'districts' in result
        assert len(result['districts']) > 0


    def test_region_analysis_with_theme_filter(self, region_data):
        """Тест анализа с фильтром по теме"""
        result = regionDistrictAnalysis(region_data, theme="Качество")

        assert result['current_theme'] == "Качество"
        assert 'stats' in result
        assert result['stats']['total'] > 0


class TestErrorHandling:
    """Тесты обработки ошибок"""


    def test_functions_with_invalid_data(self):
        """Тест функций с невалидными данными"""
        invalid_df = pd.DataFrame({'wrong_column': [1, 2, 3]})

        # Проверяем что функции не падают с ошибкой
        assert totalServed(invalid_df) == 0
        assert negativeCount(invalid_df) == 0
        assert negativeShare(invalid_df) == 0

        csi_result = csiIndex(invalid_df)
        assert csi_result == 0

        arpu_result = arpuSegments(invalid_df)
        assert isinstance(arpu_result, dict)


    @patch('lab_2.plt')
    def test_plot_functions_error_handling(self, mock_plt):
        """Тест обработки ошибок в функциях построения графиков"""
        # Настраиваем мок чтобы симулировать ошибку
        mock_plt.subplots.side_effect = Exception("Plot error")

        test_data = pd.DataFrame({
            'Тема обращения': ['Недовольство/Тест'],
            'Дата обращения': ['2024-01-01'],
            'ARPU': ['B2C Low']
        })

        # Проверяем что функции возвращают ожидаемую структуру при ошибке
        forecast_result = forecastNegativeThemes(test_data)
        assert 'plot' in forecast_result
        assert 'forecast' in forecast_result

        arpu_result = arpuNegativeThemes(test_data)
        assert 'error' in arpu_result


class TestIntegrationScenarios:
    """Интеграционные тесты"""

    @pytest.fixture
    def integration_data(self):
        """Комплексные данные для интеграционного тестирования"""
        dates = pd.date_range(start='2024-01-01', periods=15, freq='D')
        return pd.DataFrame({
            'Тема обращения': [
                'Недовольство/Качество', 'Недовольство/Тарифы', 'Поддержка',
                'Недовольство/Обслуживание', 'Консультация', 'Недовольство/Качество',
                'Недовольство/Тарифы', 'Поддержка', 'Недовольство/Обслуживание',
                'Консультация', 'Недовольство/Качество', 'Недовольство/Тарифы',
                'Поддержка', 'Недовольство/Обслуживание', 'Консультация'
            ],
            'Дата обращения': dates,
            'SA': np.random.uniform(3.0, 5.0, 15),
            'CES': np.random.uniform(3.0, 5.0, 15),
            'NPS': np.random.randint(5, 11, 15),
            'ARPU': np.random.choice(['B2C Low', 'B2C Mid', 'VIP', 'VIP adv', 'Platinum'], 15),
            'Регион': np.random.choice(['Москва', 'СПб', 'Новосибирск'], 15),
            'Район': ['Район'] * 15
        })


    def test_complete_workflow(self, integration_data):
        """Тест полного рабочего процесса"""
        # 1. Базовые метрики
        total = totalServed(integration_data)
        negative = negativeCount(integration_data)
        share = negativeShare(integration_data)

        assert total > 0
        assert negative > 0
        assert 0 <= share <= 100

        # 2. CSI расчет
        csi = csiIndex(integration_data)
        assert 0 <= csi <= 100

        # 3. ARPU анализ
        arpu_results = arpuSegments(integration_data)
        assert isinstance(arpu_results, dict)
        assert len(arpu_results) > 0

        # 4. Прогнозирование
        forecast_results = forecastNegativeThemes(integration_data, return_type='both')
        assert 'plot' in forecast_results
        assert 'forecast' in forecast_results

        # 5. Региональный анализ
        region_results = regionDistrictAnalysis(integration_data)
        assert 'table' in region_results
        assert 'stats' in region_results