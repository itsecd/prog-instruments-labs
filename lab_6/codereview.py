import hashlib
import random
import datetime
import json
from typing import List, Dict, Optional

class Transaction:
    """Класс для представления банковской транзакции"""
    
    def __init__(self, transaction_id: str, from_account: str, to_account: str, 
                 amount: float, transaction_type: str, description: str = ""):
        self.transaction_id = transaction_id
        self.from_account = from_account
        self.to_account = to_account
        self.amount = amount
        self.transaction_type = transaction_type  # 'deposit', 'withdraw', 'transfer'
        self.description = description
        self.timestamp = datetime.datetime.now()
        self.status = "pending"
    
    def execute(self) -> bool:
        """Выполнение транзакции"""
        try:
            self.status = "completed"
            return True
        except Exception as e:
            self.status = "failed"
            return False
    
    def to_dict(self) -> Dict:
        """Конвертация транзакции в словарь"""
        return {
            'transaction_id': self.transaction_id,
            'from_account': self.from_account,
            'to_account': self.to_account,
            'amount': self.amount,
            'type': self.transaction_type,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status
        }
    
    def __str__(self) -> str:
        return (f"Transaction {self.transaction_id}: {self.transaction_type} "
                f"${self.amount:.2f} from {self.from_account} to {self.to_account}")

class BankAccount:
    """Класс для представления банковского счета"""
    
    def __init__(self, account_number: str, account_holder: str, initial_balance: float = 0.0):
        self.account_number = account_number
        self.account_holder = account_holder
        self.balance = initial_balance
        self.transactions: List[Transaction] = []
        self.is_active = True
        self.created_date = datetime.datetime.now()
        self.account_type = "checking"  # 'checking', 'savings', 'business'
    
    def deposit(self, amount: float, description: str = "") -> bool:
        """Пополнение счета"""
        if amount <= 0:
            print("Сумма пополнения должна быть положительной")
            return False
        
        if not self.is_active:
            print("Счет не активен")
            return False
        
        transaction_id = f"TXN{random.randint(100000, 999999)}"
        transaction = Transaction(transaction_id, "EXTERNAL", self.account_number, 
                                 amount, "deposit", description)
        
        if transaction.execute():
            self.balance += amount
            self.transactions.append(transaction)
            print(f"Успешное пополнение: ${amount:.2f}")
            return True
        return False
    
    def withdraw(self, amount: float, description: str = "") -> bool:
        """Снятие средств"""
        if amount <= 0:
            print("Сумма снятия должна быть положительной")
            return False
        
        if amount > self.balance:
            print("Недостаточно средств на счете")
            return False
        
        if not self.is_active:
            print("Счет не активен")
            return False
        
        transaction_id = f"TXN{random.randint(100000, 999999)}"
        transaction = Transaction(transaction_id, self.account_number, "EXTERNAL",
                                 amount, "withdraw", description)
        
        if transaction.execute():
            self.balance -= amount
            self.transactions.append(transaction)
            print(f"Успешное снятие: ${amount:.2f}")
            return True
        return False
    
    def transfer(self, to_account: 'BankAccount', amount: float, description: str = "") -> bool:
        """Перевод средств на другой счет"""
        if amount <= 0:
            print("Сумма перевода должна быть положительной")
            return False
        
        if amount > self.balance:
            print("Недостаточно средств для перевода")
            return False
        
        if not self.is_active or not to_account.is_active:
            print("Один из счетов не активен")
            return False
        
        transaction_id = f"TXN{random.randint(100000, 999999)}"
        transaction = Transaction(transaction_id, self.account_number, to_account.account_number, amount, "transfer", description)
        
        if transaction.execute():
            self.balance -= amount
            to_account.balance += amount
            self.transactions.append(transaction)
            to_account.transactions.append(transaction)
            print(f"Успешный перевод: ${amount:.2f} на счет {to_account.account_number}")
            return True
        return False
    
    def get_balance(self) -> float:
        """Получение текущего баланса"""
        return self.balance
    
    def get_transaction_history(self) -> List[Transaction]:
        """Получение истории транзакций"""
        return self.transactions
    
    def get_account_info(self) -> Dict:
        """Получение информации о счете"""
        return {
            'account_number': self.account_number,
            'account_holder': self.account_holder,
            'balance': self.balance,
            'account_type': self.account_type,
            'is_active': self.is_active,
            'created_date': self.created_date.isoformat(),
            'total_transactions': len(self.transactions)
        }
    
    def deactivate(self) -> bool:
        """Деактивация счета"""
        if self.balance == 0:
            self.is_active = False
            return True
        print("Нельзя деактивировать счет с ненулевым балансом")
        return False
    
    def __str__(self) -> str:
        status = "активен" if self.is_active else "не активен"
        return (f"Счет {self.account_number} ({self.account_holder}): "
                f"${self.balance:.2f}, {status}")

class Bank:
    """Класс для представления банка"""
    
    def __init__(self, bank_name: str):
        self.bank_name = bank_name
        self.accounts: Dict[str, BankAccount] = {}
        self.total_transactions = 0
    
    def create_account(self, account_holder: str, initial_deposit: float = 0.0) -> BankAccount:
        """Создание нового счета"""
        account_number = f"ACC{random.randint(10000000, 99999999)}"
        
        # Проверка на уникальность номера счета
        while account_number in self.accounts:
            account_number = f"ACC{random.randint(10000000, 99999999)}"
        
        new_account = BankAccount(account_number, account_holder, initial_deposit)
        self.accounts[account_number] = new_account
        
        if initial_deposit > 0:
            new_account.deposit(initial_deposit, "Первоначальный взнос")
        
        print(f"Создан новый счет {account_number} для {account_holder}")
        return new_account
    
    def get_account(self, account_number: str) -> Optional[BankAccount]:
        """Получение счета по номеру"""
        return self.accounts.get(account_number)
    
    def close_account(self, account_number: str) -> bool:
        """Закрытие счета"""
        account = self.get_account(account_number)
        if account:
            if account.deactivate():
                del self.accounts[account_number]
                print(f"Счет {account_number} закрыт")
                return True
        else:
            print(f"Счет {account_number} не найден")
        return False
    
    def get_total_deposits(self) -> float:
        """Общая сумма депозитов в банке"""
        return sum(account.balance for account in self.accounts.values())
    
    def get_bank_statistics(self) -> Dict:
        """Статистика банка"""
        total_accounts = len(self.accounts)
        active_accounts = sum(1 for account in self.accounts.values() if account.is_active)
        total_transactions = sum(len(account.transactions) for account in self.accounts.values())
        
        return {
            'bank_name': self.bank_name,
            'total_accounts': total_accounts,
            'active_accounts': active_accounts,
            'total_deposits': self.get_total_deposits(),
            'total_transactions': total_transactions
        }
    
    def find_accounts_by_holder(self, account_holder: str) -> List[BankAccount]:
        """Поиск счетов по владельцу"""
        return [account for account in self.accounts.values() 
                if account.account_holder.lower() == account_holder.lower()]
    
    def save_to_file(self, filename: str) -> bool:
        """Сохранение данных банка в файл"""
        data = {
            'bank_name': self.bank_name,
            'accounts': {
                acc_num: {
                    'account_info': account.get_account_info(),
                    'transactions': [t.to_dict() for t in account.transactions]
                }
                for acc_num, account in self.accounts.items()
            }
        }
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Данные банка сохранены в {filename}")
        return True

def DemoBankSystem():
    """Демонстрация работы банковской системы"""
    
    # Создание банка
    bank = Bank("ТехноБанк")
    
    # Создание счетов
    print("=" * 50)
    print("СОЗДАНИЕ СЧЕТОВ")
    print("=" * 50)
    
    account1 = bank.create_account("Иван Петров", 1000.0)
    account2 = bank.create_account("Мария Сидорова", 500.0)
    account3 = bank.create_account("Алексей Иванов", 200.0)
    
    # Операции со счетами
    print("\n" + "=" * 50)
    print("БАНКОВСКИЕ ОПЕРАЦИИ")
    print("=" * 50)
    
    # Пополнения
    account1.deposit(300.0, "Зарплата")
    account2.deposit(150.0, "Возврат долга")
    
    # Снятия
    account1.withdraw(200.0, "Наличные")
    account3.withdraw(50.0, "Покупки")
    
    # Переводы
    account1.transfer(account2, 100.0, "Перевод другу")
    account2.transfer(account3, 75.0, "Оплата услуг")
    
    # Пытаемся перевести больше чем есть
    account3.transfer(account1, 500.0, "Большой перевод")  # Должно не сработать
    
    # Информация о счетах
    print("\n" + "=" * 50)
    print("ИНФОРМАЦИЯ О СЧЕТАХ")
    print("=" * 50)
    
    for account in [account1, account2, account3]:
        info = account.get_account_info()
        print(f"\nВладелец: {info['account_holder']}")
        print(f"Номер счета: {info['account_number']}")
        print(f"Баланс: ${info['balance']:.2f}")
        print(f"Кол-во транзакций: {info['total_transactions']}")
        print(f"Статус: {'активен' if info['is_active'] else 'не активен'}")
    
    # Статистика банка
    print("\n" + "=" * 50)
    print("СТАТИСТИКА БАНКА")
    print("=" * 50)
    
    stats = bank.get_bank_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: ${value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # История транзакций для первого счета
    print("\n" + "=" * 50)
    print(f"ИСТОРИЯ ТРАНЗАКЦИЙ ДЛЯ {account1.account_number}")
    print("=" * 50)
    
    for i, transaction in enumerate(account1.get_transaction_history(), 1):
        trans_dict = transaction.to_dict()
        print(f"{i}. {trans_dict['timestamp'][:16]} | "
              f"{trans_dict['type'].upper():<10} | "
              f"${trans_dict['amount']:>8.2f} | "
              f"{trans_dict['description']}")
    
    # Сохранение данных
    print("\n" + "=" * 50)
    print("СОХРАНЕНИЕ ДАННЫХ")
    print("=" * 50)
    
    bank.save_to_file("bank_data.json")
    
    print("\nДемонстрация завершена!")

if __name__ == "__main__":
    DemoBankSystem()