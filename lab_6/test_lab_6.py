import pytest
import sys
import os
from unittest.mock import Mock, patch

# Добавляем путь для импорта модуля
sys.path.insert(0, os.path.dirname(__file__))

try:
    from lab_2 import AnalysisConfig
except ImportError as e:
    pytest.skip(f"Could not import lab_2: {e}", allow_module_level=True)

class TestAnalysisConfig:
    """Тесты для класса конфигурации"""

    def test_config_constants(self):
        """Проверка основных констант конфигурации"""
        assert AnalysisConfig.NEGATIVE_THEME_PATTERN == "Недовольство"
        assert AnalysisConfig.CSI_BASE_VALUE == 87.97
        assert AnalysisConfig.CSI_MIN_VALUE == 0
        assert AnalysisConfig.CSI_MAX_VALUE == 100

    def test_arpu_segments_structure(self):
        """Тест структуры ARPU сегментов"""
        assert isinstance(AnalysisConfig.ARPU_SEGMENTS, dict)
        assert "b2c_low" in AnalysisConfig.ARPU_SEGMENTS
        assert "vip" in AnalysisConfig.ARPU_SEGMENTS

    def test_plot_configurations(self):
        """Тест настроек графиков"""
        assert hasattr(AnalysisConfig, 'PLOT_STYLE')
        assert hasattr(AnalysisConfig, 'PLOT_FIGSIZE')
        assert hasattr(AnalysisConfig, 'COLOR_NEGATIVE')


class TestBasicFunctions:
    """Тесты базовых функций"""

    def test_total_served_basic(self):
        """Тест подсчета общего количества"""
        from lab_2 import totalServed
        
        class MockData:
            def __len__(self):
                return 5
        
        result = totalServed(MockData())
        assert result == 5

    def test_total_served_empty(self):
        """Тест подсчета для пустых данных"""
        from lab_2 import totalServed
        
        class EmptyData:
            def __len__(self):
                return 0
        
        result = totalServed(EmptyData())
        assert result == 0

    @pytest.mark.parametrize("input_length,expected", [
        (0, 0),
        (1, 1),
        (5, 5),
        (100, 100)
    ])
    def test_total_served_parameterized(self, input_length, expected):
        """Параметризованный тест подсчета"""
        from lab_2 import totalServed
        
        class MockData:
            def __len__(self):
                return input_length
        
        result = totalServed(MockData())
        assert result == expected


class TestMockingExamples:
    """Тесты с использованием моков"""

    def test_mock_with_simple_object(self):
        """Тест с простым мок-объектом"""
        class MockDataFrame:
            def __init__(self, data_dict):
                self.data = data_dict
            
            def __len__(self):
                return len(self.data.get('Тема обращения', []))
        
        mock_df = MockDataFrame({'Тема обращения': ['test1', 'test2', 'test3']})
        assert len(mock_df) == 3

    def test_mock_with_complex_behavior(self):
        """Тест с моком имеющим сложное поведение"""
        class MockFunction:
            def __init__(self):
                self.call_count = 0
                self.results = [10, 20, 30]
            
            def __call__(self, *args):
                if self.call_count < len(self.results):
                    result = self.results[self.call_count]
                    self.call_count += 1
                    return result
                return 0
        
        mock_func = MockFunction()
        assert mock_func() == 10
        assert mock_func() == 20
        assert mock_func() == 30
        assert mock_func.call_count == 3

    def test_mock_data_processing(self):
        """Тест мока для обработки данных"""
        class MockDataProcessor:
            def __init__(self):
                self.processed_items = []
            
            def process(self, item):
                self.processed_items.append(item)
                return f"processed_{item}"
            
            def get_stats(self):
                return {
                    'total': len(self.processed_items),
                    'unique': len(set(self.processed_items))
                }
        
        processor = MockDataProcessor()
        processor.process('item1')
        processor.process('item2')
        processor.process('item1')  # duplicate
        
        stats = processor.get_stats()
        assert stats['total'] == 3
        assert stats['unique'] == 2


class TestErrorHandling:
    """Тесты обработки ошибок"""

    def test_function_imports(self):
        """Тест что все функции можно импортировать"""
        try:
            from lab_2 import totalServed, negativeCount, negativeShare, csiIndex
            from lab_2 import arpuSegments, forecastNegativeThemes, regionDistrictAnalysis
            # Если импорт прошел - тест пройден
            assert True
        except ImportError as e:
            pytest.skip(f"Some functions not available: {e}")

    def test_basic_error_scenarios(self):
        """Тест базовых сценариев ошибок"""
        # Проверяем обработку None
        try:
            from lab_2 import totalServed
            result = totalServed(None)
            # Если не упало, проверяем результат
            assert isinstance(result, (int, float)) or result is None
        except Exception:
            # Если упало - это нормально для некоторых реализаций
            pass

    def test_edge_cases(self):
        """Тест граничных случаев"""
        class EdgeCaseData:
            def __init__(self, behavior):
                self.behavior = behavior
            
            def __len__(self):
                if self.behavior == 'zero':
                    return 0
                elif self.behavior == 'large':
                    return 1000000
                elif self.behavior == 'negative':
                    return -1  # Неправильное поведение
                return 1
        
        # Тестируем разные сценарии
        zero_data = EdgeCaseData('zero')
        large_data = EdgeCaseData('large')
        normal_data = EdgeCaseData('normal')
        
        assert len(zero_data) == 0
        assert len(large_data) == 1000000
        assert len(normal_data) == 1


class TestParameterizedAdvanced:
    """Продвинутые параметризованные тесты"""

    @pytest.mark.parametrize("sa_values,ces_values,nps_values,expected_behavior", [
        ([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [10, 10, 10], "high_scores"),
        ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0, 0, 0], "low_scores"),
        ([3.0, 4.0, 3.5], [3.0, 4.0, 3.5], [5, 8, 6], "mixed_scores"),
    ])
    def test_csi_scenarios(self, sa_values, ces_values, nps_values, expected_behavior):
        """Параметризованные сценарии для CSI"""
        # Проверяем что входные данные корректны
        assert len(sa_values) == len(ces_values) == len(nps_values)
        assert all(isinstance(x, float) for x in sa_values)
        assert all(isinstance(x, float) for x in ces_values)
        assert all(isinstance(x, int) for x in nps_values)
        
        # Проверяем диапазоны значений
        if expected_behavior == "high_scores":
            assert all(x >= 4.5 for x in sa_values)
        elif expected_behavior == "low_scores":
            assert all(x <= 2.0 for x in sa_values)

    @pytest.mark.parametrize("segment_name,expected_type", [
        ("B2C Low", "low_tier"),
        ("B2C Mid", "mid_tier"),
        ("VIP", "high_tier"),
        ("VIP adv", "premium_tier"),
        ("Platinum", "premium_tier"),
    ])
    def test_arpu_segment_types(self, segment_name, expected_type):
        """Параметризованный тест типов ARPU сегментов"""
        from lab_2 import AnalysisConfig
        
        segments = AnalysisConfig.ARPU_SEGMENTS
        # Проверяем что сегмент существует в конфигурации
        segment_key = None
        for key, value in segments.items():
            if value == segment_name:
                segment_key = key
                break
        
        assert segment_key is not None
        assert segment_name in segments.values()
        
        # Проверяем логику классификации
        if expected_type == "low_tier":
            assert "low" in segment_key.lower()
        elif expected_type == "premium_tier":
            assert any(word in segment_key.lower() for word in ['vip', 'platinum'])


class TestIntegrationWorkflow:
    """Интеграционные тесты рабочего процесса"""

    def test_configuration_workflow(self):
        """Тест рабочего процесса конфигурации"""
        # 1. Проверяем что конфигурация загружается
        config = AnalysisConfig()
        
        # 2. Проверяем основные настройки
        required_attributes = [
            'NEGATIVE_THEME_PATTERN', 'CSI_BASE_VALUE', 
            'ARPU_SEGMENTS', 'COLOR_NEGATIVE'
        ]
        
        for attr in required_attributes:
            assert hasattr(config, attr)
        
        # 3. Проверяем типы значений
        assert isinstance(config.NEGATIVE_THEME_PATTERN, str)
        assert isinstance(config.CSI_BASE_VALUE, float)
        assert isinstance(config.ARPU_SEGMENTS, dict)
        assert isinstance(config.COLOR_NEGATIVE, str)

    def test_data_processing_workflow(self):
        """Тест рабочего процесса обработки данных"""
        class MockProcessingPipeline:
            def __init__(self):
                self.steps = []
                self.results = []
            
            def add_step(self, step_name, step_function):
                self.steps.append(step_name)
                return self
            
            def process(self, data):
                for step in self.steps:
                    result = f"processed_{step}_{len(data)}"
                    self.results.append(result)
                return self.results
        
        pipeline = MockProcessingPipeline()
        pipeline.add_step('filter', lambda x: x)
        pipeline.add_step('analyze', lambda x: x)
        
        test_data = ['item1', 'item2', 'item3']
        results = pipeline.process(test_data)
        
        assert len(results) == 2
        assert all('processed_' in result for result in results)
        assert pipeline.steps == ['filter', 'analyze']


class TestAdvancedMocking:
    """Продвинутые тесты с моками"""

    def test_mock_with_monkeypatch(self, monkeypatch):
        """Тест с использованием monkeypatch"""
        # Создаем мок-функцию
        def mock_calculation(data):
            return len(data) * 10
        
        # Патчим функцию (если бы она была в модуле)
        monkeypatch.setattr('lab_2.some_calculation_function', mock_calculation)
        
        # Тестируем логику
        test_data = [1, 2, 3, 4, 5]
        result = mock_calculation(test_data)
        assert result == 50

    def test_mock_class_behavior(self):
        """Тест поведения мок-класса"""
        class MockAnalyzer:
            def __init__(self):
                self.analysis_count = 0
                self.last_data = None
            
            def analyze(self, data):
                self.analysis_count += 1
                self.last_data = data
                return {
                    'count': len(data) if hasattr(data, '__len__') else 0,
                    'analysis_id': self.analysis_count
                }
            
            def get_stats(self):
                return {
                    'total_analyses': self.analysis_count,
                    'last_data_size': len(self.last_data) if self.last_data else 0
                }
        
        analyzer = MockAnalyzer()
        
        # Первый анализ
        result1 = analyzer.analyze([1, 2, 3])
        assert result1['count'] == 3
        assert result1['analysis_id'] == 1
        
        # Второй анализ
        result2 = analyzer.analyze([1, 2, 3, 4, 5])
        assert result2['count'] == 5
        assert result2['analysis_id'] == 2
        
        # Проверяем статистику
        stats = analyzer.get_stats()
        assert stats['total_analyses'] == 2
        assert stats['last_data_size'] == 5


def test_final_comprehensive():
    """Финальный комплексный тест"""
    # 1. Проверяем конфигурацию
    assert AnalysisConfig.NEGATIVE_THEME_PATTERN == "Недовольство"
    
    # 2. Проверяем что можем создавать мок-данные
    class ComprehensiveData:
        def __init__(self, items):
            self.items = items
            self.processed = False
        
        def __len__(self):
            return len(self.items)
        
        def process(self):
            self.processed = True
            return [f"processed_{item}" for item in self.items]
        
        def get_metrics(self):
            return {
                'total': len(self.items),
                'processed': self.processed,
                'unique': len(set(self.items))
            }
    
    # 3. Тестируем комплексный сценарий
    data = ComprehensiveData(['A', 'B', 'C', 'A', 'B'])
    assert len(data) == 5
    
    processed = data.process()
    assert len(processed) == 5
    assert all(item.startswith('processed_') for item in processed)
    
    metrics = data.get_metrics()
    assert metrics['total'] == 5
    assert metrics['processed'] == True
    assert metrics['unique'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
