import json
import os
import pytest
import tempfile
from unittest.mock import Mock


class TestModels:
    """Data Model tests"""

    def test_bank_config_creation(self):
        from models import BankConfig
        config = BankConfig("https://example.com", "Test Bank", "debitcards")
        assert config.url == "https://example.com"
        assert config.name == "Test Bank"
        assert config.product_type == "debitcards"

    def test_card_data_creation(self):
        from models import CardData
        card = CardData("https://example.com/card", name="Test Card", bank="Test Bank")
        assert card.url == "https://example.com/card"
        assert card.name == "Test Card"
        assert card.bank == "Test Bank"


class TestUtils:
    """Utility tests"""

    def test_save_results_to_file(self):
        from utils import save_results_to_file

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            test_data = [{'name': 'Test Card', 'bank': 'Test Bank'}]
            save_results_to_file(test_data, temp_path)

            assert os.path.exists(temp_path)
            with open(temp_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            assert saved_data == test_data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestExtractors:
    """Data Parser tests"""

    @pytest.mark.parametrize("html_content,expected_count", [
        ("", 0),
        ("<div>No scripts</div>", 0),
        ('<script type="application/ld+json">{"@type": "Product"}</script>', 1),
        ('<script type="application/ld+json">{"test": 1}</script>', 1)
    ])
    def test_json_ld_extractor_various_inputs(self, html_content, expected_count):
        from extractors import JsonLdExtractor
        extractor = JsonLdExtractor()
        result = extractor.extract_from_html(html_content)
        assert len(result) == expected_count


class TestProcessors:
    """Processor tests (with mock)"""

    def test_card_processor_success_with_mocks(self, mocker):
        from processors import CardProcessor

        mock_response = mocker.Mock()
        mock_response.text = "<html>test content</html>"
        mocker.patch('cffi_requests.get', return_value=mock_response)
        mocker.patch('processors.DataModuleExtractor.extract',
                     return_value={'data': {'cardName': 'Test Card', 'bankName': 'Test Bank'}})
        mocker.patch('processors.DataCleaner.clean',
                     return_value={'name': 'Test Card', 'bank': 'Test Bank'})

        processor = CardProcessor(delay=0)
        result = processor.process_single_card("https://example.com/card", "debitcards")

        assert result['success'] is True
        assert result['name'] == 'Test Card'
        assert result['bank'] == 'Test Bank'

    def test_card_processor_error_with_mocks(self, mocker):
        from processors import CardProcessor

        mocker.patch('cffi_requests.get', side_effect=Exception("Network error"))

        processor = CardProcessor(delay=0)
        result = processor.process_single_card("https://example.com/card", "debitcards")

        assert result['success'] is False
        assert 'error' in result
