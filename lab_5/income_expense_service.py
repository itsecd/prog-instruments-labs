from datetime import datetime

class Transaction:
    def __init__(self, amount: float, description: str, transaction_type: str, date: datetime = None):
        """Инициализация транзакции.

        Args:
            amount (float): Сумма транзакции. Должна быть положительной.
            description (str): Описание транзакции.
            transaction_type (str): Тип транзакции ("income" или "expense").
            date (datetime, optional): Дата транзакции. По умолчанию текущая дата.
        """
        if transaction_type not in {"income", "expense"}:
            raise ValueError("Transaction type must be 'income' or 'expense'.")
        if amount <= 0:
            raise ValueError("Amount must be positive.")

        self.amount = amount
        self.description = description
        self.transaction_type = transaction_type
        self.date = date or datetime.now()

    def __repr__(self):
        date_str = self.date.strftime("%Y-%m-%d")
        return f"{self.transaction_type.capitalize()}: {self.amount:.2f} ({self.description}) on {date_str}"


class Account:
    def __init__(self, name: str, balance: float = 0.0):
        """Инициализация счета.

        Args:
            name (str): Название счета.
            balance (float, optional): Начальный баланс. По умолчанию 0.0.
        """
        self.name = name
        self.balance = balance

    def adjust_balance(self, amount: float):
        """Изменить баланс счета.

        Args:
            amount (float): Сумма изменения (может быть отрицательной).
        """
        self.balance += amount

    def __repr__(self):
        return f"Account(name={self.name}, balance={self.balance:.2f})"


class BudgetService:
    def __init__(self):
        """Инициализация сервиса управления бюджетом."""
        self.transactions = []
        self.accounts = {}

    def add_account(self, account: Account):
        """Добавить новый счет.

        Args:
            account (Account): Объект счета.
        """
        if account.name in self.accounts:
            raise ValueError(f"Account with name '{account.name}' already exists.")
        self.accounts[account.name] = account

    def add_transaction(self, transaction: Transaction, account_name: str):
        """Добавить транзакцию к указанному счету.

        Args:
            transaction (Transaction): Объект транзакции.
            account_name (str): Название счета.
        """
        if account_name not in self.accounts:
            raise ValueError(f"Account '{account_name}' does not exist.")

        account = self.accounts[account_name]
        if transaction.transaction_type == "expense" and account.balance < transaction.amount:
            raise ValueError("Insufficient funds for this transaction.")

        adjustment = transaction.amount if transaction.transaction_type == "income" else -transaction.amount
        account.adjust_balance(adjustment)
        self.transactions.append((transaction, account_name))

    def edit_transaction(self, transaction_index: int, new_transaction: Transaction, new_account_name: str):
        """Редактировать существующую транзакцию.

        Args:
            transaction_index (int): Индекс транзакции в списке.
            new_transaction (Transaction): Новая версия транзакции.
            new_account_name (str): Название нового счета для транзакции.
        """
        if transaction_index < 0 or transaction_index >= len(self.transactions):
            raise IndexError("Transaction index out of range.")

        old_transaction, old_account_name = self.transactions[transaction_index]

        # Возврат средств на старый счет
        old_account = self.accounts[old_account_name]
        adjustment = old_transaction.amount if old_transaction.transaction_type == "income" else -old_transaction.amount
        old_account.adjust_balance(-adjustment)

        # Проверка и списание/зачисление на новый счет
        if new_account_name not in self.accounts:
            raise ValueError(f"Account '{new_account_name}' does not exist.")

        new_account = self.accounts[new_account_name]
        if new_transaction.transaction_type == "expense" and new_account.balance < new_transaction.amount:
            raise ValueError("Insufficient funds for this transaction.")

        adjustment = new_transaction.amount if new_transaction.transaction_type == "income" else -new_transaction.amount
        new_account.adjust_balance(adjustment)

        # Обновление транзакции
        self.transactions[transaction_index] = (new_transaction, new_account_name)

    def get_balance(self) -> float:
        """Получить общий баланс по всем счетам.

        Returns:
            float: Общий баланс.
        """
        return sum(account.balance for account in self.accounts.values())

    def get_transaction_summary(self):
        """Получить сводку всех транзакций.

        Returns:
            str: Сводка транзакций в читаемом формате.
        """
        if not self.transactions:
            return "No transactions available."

        summary = [f"{transaction} on {account_name}" for transaction, account_name in self.transactions]
        return "\n".join(summary)

    def get_transactions_by_type(self, transaction_type: str):
        """Получить список транзакций определенного типа.

        Args:
            transaction_type (str): Тип транзакции ("income" или "expense").

        Returns:
            list[Transaction]: Список транзакций данного типа.
        """
        if transaction_type not in {"income", "expense"}:
            raise ValueError("Transaction type must be 'income' or 'expense'.")

        return [t for t, _ in self.transactions if t.transaction_type == transaction_type]

    def get_transactions_by_date_range(self, start_date: datetime, end_date: datetime):
        """Получить транзакции за определенный период времени.

        Args:
            start_date (datetime): Начальная дата.
            end_date (datetime): Конечная дата.

        Returns:
            list[Transaction]: Список транзакций за указанный период.
        """
        return [t for t, _ in self.transactions if start_date <= t.date <= end_date]

    def get_monthly_report(self, year: int, month: int):
        """Получить отчет за указанный месяц.

        Args:
            year (int): Год.
            month (int): Месяц.

        Returns:
            str: Отчет по доходам, расходам и балансу за месяц.
        """
        monthly_transactions = [t for t, _ in self.transactions if t.date.year == year and t.date.month == month]
        income = sum(t.amount for t in monthly_transactions if t.transaction_type == "income")
        expenses = sum(t.amount for t in monthly_transactions if t.transaction_type == "expense")
        balance = income - expenses

        return (
            f"Report for {year}-{month:02d}:\n"
            f"Total Income: {income:.2f}\n"
            f"Total Expenses: {expenses:.2f}\n"
            f"Balance: {balance:.2f}"
        )


# Пример использования
if __name__ == "__main__":
    budget = BudgetService()

    account1 = Account("Main Account", 1000)
    account2 = Account("Savings", 500)
    budget.add_account(account1)
    budget.add_account(account2)

    budget.add_transaction(Transaction(1000, "Salary", "income", datetime(2024, 12, 1)), "Main Account")
    budget.add_transaction(Transaction(200, "Groceries", "expense", datetime(2024, 12, 3)), "Main Account")
    budget.add_transaction(Transaction(150, "Internet", "expense", datetime(2024, 12, 5)), "Main Account")
    budget.add_transaction(Transaction(500, "Freelance", "income", datetime(2024, 12, 10)), "Savings")

    print("Balance:", budget.get_balance())
    print("\nTransactions:")
    print(budget.get_transaction_summary())

    print("\nMonthly Report:")
    print(budget.get_monthly_report(2024, 12))

    print("\nTransactions by Type (Income):")
    print(budget.get_transactions_by_type("income"))

    print("\nTransactions in Date Range (2024-12-01 to 2024-12-05):")
    print(budget.get_transactions_by_date_range(datetime(2024, 12, 1), datetime(2024, 12, 5)))
