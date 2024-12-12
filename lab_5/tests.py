import pytest
from unittest.mock import MagicMock
from datetime import datetime
from income_expense_service import Account, BudgetService, Transaction

@pytest.mark.parametrize("account_name, initial_balance, transaction_amount, transaction_type, expected_balance", [
    ("Account1", 1000, 200, "expense", 800),
    ("Account2", 500, 100, "income", 600),
    ("Account3", 300, 300, "expense", 0)
])
def test_add_transaction_to_different_accounts(account_name, initial_balance, transaction_amount, transaction_type, expected_balance):
    budget = BudgetService()
    account = Account(account_name, initial_balance)
    budget.add_account(account)

    transaction = Transaction(transaction_amount, "Test Transaction", transaction_type)
    budget.add_transaction(transaction, account_name)

    assert budget.accounts[account_name].balance == expected_balance


def test_add_account():
    budget = BudgetService()
    account = Account("Test Account", 500)
    budget.add_account(account)

    assert "Test Account" in budget.accounts
    assert budget.accounts["Test Account"].balance == 500

def test_add_transaction():
    budget = BudgetService()
    account = Account("Main", 1000)
    budget.add_account(account)

    transaction = Transaction(200, "Groceries", "expense")
    budget.add_transaction(transaction, "Main")

    assert budget.get_balance() == 800
    assert len(budget.transactions) == 1

def test_insufficient_funds():
    budget = BudgetService()
    account = Account("Main", 100)
    budget.add_account(account)

    transaction = Transaction(200, "Expensive Item", "expense")
    with pytest.raises(ValueError, match="Insufficient funds"):
        budget.add_transaction(transaction, "Main")

def test_edit_transaction():
    budget = BudgetService()
    account = Account("Main", 1000)
    budget.add_account(account)

    transaction = Transaction(100, "Internet", "expense")
    budget.add_transaction(transaction, "Main")

    new_transaction = Transaction(50, "Phone Bill", "expense")
    budget.edit_transaction(0, new_transaction, "Main")

    assert budget.get_balance() == 950
    assert budget.transactions[0][0].description == "Phone Bill"

def test_get_monthly_report():
    budget = BudgetService()
    account = Account("Main", 1000)
    budget.add_account(account)

    budget.add_transaction(Transaction(1000, "Salary", "income", datetime(2024, 12, 1)), "Main")
    budget.add_transaction(Transaction(200, "Groceries", "expense", datetime(2024, 12, 3)), "Main")

    report = budget.get_monthly_report(2024, 12)
    assert "Total Income: 1000.00" in report
    assert "Total Expenses: 200.00" in report

def test_get_transactions_by_type():
    budget = BudgetService()
    account = Account("Main", 1000)
    budget.add_account(account)

    budget.add_transaction(Transaction(1000, "Salary", "income"), "Main")
    budget.add_transaction(Transaction(200, "Groceries", "expense"), "Main")

    income_transactions = budget.get_transactions_by_type("income")
    assert len(income_transactions) == 1
    assert income_transactions[0].description == "Salary"

def test_get_transactions_by_date_range():
    budget = BudgetService()
    account = Account("Main", 1000)
    budget.add_account(account)

    budget.add_transaction(Transaction(1000, "Salary", "income", datetime(2024, 12, 1)), "Main")
    budget.add_transaction(Transaction(200, "Groceries", "expense", datetime(2024, 12, 3)), "Main")
    budget.add_transaction(Transaction(150, "Internet", "expense", datetime(2024, 11, 20)), "Main")

    date_range_transactions = budget.get_transactions_by_date_range(datetime(2024, 12, 1), datetime(2024, 12, 31))
    assert len(date_range_transactions) == 2
    assert all(t.date.month == 12 for t in date_range_transactions)

def test_add_account_mock():
    budget = BudgetService()
    budget.add_account = MagicMock()

    account = Account("Mock Account", 1000)
    budget.add_account(account)

    # Проверяем, что метод add_account был вызван с правильным аргументом
    budget.add_account.assert_called_once_with(account)

