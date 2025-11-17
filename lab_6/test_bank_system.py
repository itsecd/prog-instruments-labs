from codereview import BankAccount, Bank, Transaction
from datetime import datetime
import pytest
from unittest.mock import patch, mock_open


class TestTransaction:
    def setup_method(self):
        self.transaction = Transaction("ID", "ACC1", "ACC2", 100.0, "transfer", "Test transfer")
    

    def test_transaction_creation(self):
        assert self.transaction.transaction_id == "ID"
        assert self.transaction.from_account == "ACC1"
        assert self.transaction.to_account == "ACC2"
        assert self.transaction.amount == 100.0
        assert self.transaction.transaction_type == "transfer"
        assert self.transaction.description == "Test transfer"
        assert self.transaction.status == "pending"
        assert isinstance(self.transaction.timestamp, datetime)
    

    def test_transaction_success(self):
        result = self.transaction.execute()
        
        assert result == True
        assert self.transaction.status == "completed"
    

    def test_transaction_to_dict(self):
        result = self.transaction.to_dict()
        
        assert result['transaction_id'] == "ID"
        assert result['from_account'] == "ACC1"
        assert result['to_account'] == "ACC2"
        assert result['amount'] == 100.0
        assert result['type'] == "transfer"
        assert result['description'] == "Test transfer"
        assert result['status'] == "pending"
        assert 'timestamp' in result
    

    def test_transaction_str(self):
        result = str(self.transaction)
        
        assert "ID" in result
        assert "ACC1" in result
        assert "ACC2" in result
        assert "transfer" in result
        assert "100.00" in result


class TestBankAccount:
    def setup_method(self):
        self.account = BankAccount("123456", "Ivan Ivanov", 0.0)
    

    def test_account_creation(self):
        assert self.account.account_number == "123456"
        assert self.account.account_holder == "Ivan Ivanov"
        assert self.account.balance == 0.0
        assert self.account.is_active == True
        assert self.account.account_type == "checking"
        assert len(self.account.transactions) == 0
        assert isinstance(self.account.created_date, datetime)
    

    @pytest.mark.parametrize("initial_balance,deposit_amount,expected_balance,should_succeed", [
        (1000.0, 500.0, 1500.0, True),
        (1000.0, 0.0, 1000.0, False),
        (1000.0, -100.0, 1000.0, False),
    ])
    def test_deposit_various_amounts(self, initial_balance, deposit_amount, expected_balance, should_succeed):
        """Параметризованный тест пополнения"""
        self.account.balance = initial_balance
        
        with patch('builtins.print') as mock_print:
            result = self.account.deposit(deposit_amount, "Test deposit")
            
            assert result == should_succeed
            assert self.account.balance == expected_balance
            
            if not should_succeed:
                mock_print.assert_called()
    

    def test_deposit_on_inactive_account(self):
        self.account.is_active = False
        
        with patch('builtins.print') as mock_print:
            result = self.account.deposit(500.0)
            
            assert result == False
            assert self.account.balance == 0.0
            mock_print.assert_called_with("Счет не активен")


class TestBank:    
    def setup_method(self):
        self.bank = Bank("Name")
    

    def test_bank_creation(self):
        assert self.bank.bank_name == "Name"
        assert self.bank.accounts == {}
        assert self.bank.total_transactions == 0
    
    
    def test_save_to_file(self):
        self.bank.create_account("Ivan Ivanov", 0.0)
        
        mock_file = mock_open()
        
        with patch('builtins.open', mock_file), \
             patch('json.dump') as mock_json_dump, \
             patch('builtins.print') as mock_print:
            
            result = self.bank.save_to_file('test_bank_data.json')
            
            assert result == True
            mock_file.assert_called_once_with('test_bank_data.json', 'w', encoding='utf-8')
            mock_json_dump.assert_called_once()
            mock_print.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])