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