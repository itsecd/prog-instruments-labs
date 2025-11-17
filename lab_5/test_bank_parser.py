import tempfile
import os
import json


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