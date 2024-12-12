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
    """
    Тест для проверки добавления транзакции в различные аккаунты.
    
    Параметры:
    - account_name: str — имя аккаунта.
    - initial_balance: float — начальный баланс аккаунта.
    - transaction_amount: float — сумма транзакции.
    - transaction_type: str — тип транзакции (доход/расход).
    - expected_balance: float — ожидаемый баланс после транзакции.
    """
    budget = BudgetService()
    account = Account(account_name, initial_balance)
    budget.add_account(account)

    transaction = Transaction(transaction_amount, "Test Transaction", transaction_type)
    budget.add_transaction(transaction, account_name)

    assert budget.accounts[account_name].balance == expected_balance


def test_add_account():
    """
    Тест для проверки добавления нового аккаунта.
    """
    budget = BudgetService()
    account = Account("Test Account", 500)
    budget.add_account(account)

    assert "Test Account" in budget.accounts
    assert budget.accounts["Test Account"].balance == 500

def test_add_transaction():
    """
    Тест для проверки добавления транзакции на существующий аккаунт.
    """
    budget = BudgetService()
    account = Account("Main", 1000)
    budget.add_account(account)

    transaction = Transaction(200, "Groceries", "expense")
    budget.add_transaction(transaction, "Main")

    assert budget.get_balance() == 800
    assert len(budget.transactions) == 1

def test_insufficient_funds():
    """
    Тест для проверки случая недостаточных средств при добавлении транзакции.
    """
    budget = BudgetService()
    account = Account("Main", 100)
    budget.add_account(account)

    transaction = Transaction(200, "Expensive Item", "expense")
    with pytest.raises(ValueError, match="Insufficient funds"):
        budget.add_transaction(transaction, "Main")

def test_edit_transaction():
    """
    Тест для проверки редактирования существующей транзакции.
    """
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
    """
    Тест для генерации ежемесячного отчета о доходах и расходах.
    """
    budget = BudgetService()
    account = Account("Main", 1000)
    budget.add_account(account)

    budget.add_transaction(Transaction(1000, "Salary", "income", datetime(2024, 12, 1)), "Main")
    budget.add_transaction(Transaction(200, "Groceries", "expense", datetime(2024, 12, 3)), "Main")

    report = budget.get_monthly_report(2024, 12)
    assert "Total Income: 1000.00" in report
    assert "Total Expenses: 200.00" in report

def test_get_transactions_by_type():
    """
    Тест для получения транзакций по типу (доход/расход).
    """
    budget = BudgetService()
    account = Account("Main", 1000)
    budget.add_account(account)

    budget.add_transaction(Transaction(1000, "Salary", "income"), "Main")
    budget.add_transaction(Transaction(200, "Groceries", "expense"), "Main")

    income_transactions = budget.get_transactions_by_type("income")
    assert len(income_transactions) == 1
    assert income_transactions[0].description == "Salary"

def test_get_transactions_by_date_range():
    """
    Тест для получения транзакций по диапазону дат.
    """
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
    """
    Тест с использованием mock-объекта для проверки вызова метода добавления аккаунта.
    """
    budget = BudgetService()
    budget.add_account = MagicMock()

    account = Account("Mock Account", 1000)
    budget.add_account(account)

    # Проверяем, что метод add_account был вызван с правильным аргументом
    budget.add_account.assert_called_once_with(account)
     