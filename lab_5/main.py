import datetime
import random
import logging
import logging.config
import json
from typing import List, Dict, Optional


def setup_logging():
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            },
            'simple': {
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file_handler': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'bank_system.log',
                'encoding': 'utf-8',
                'mode': 'w'
            },
            'error_handler': {
                'class': 'logging.FileHandler',
                'level': 'WARNING',
                'formatter': 'detailed',
                'filename': 'bank_errors.log',
                'encoding': 'utf-8',
                'mode': 'w'
            }
        },
        'loggers': {
            'bank_system': {
                'level': 'DEBUG',
                'handlers': ['console', 'file_handler', 'error_handler'],
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    return logging.getLogger('bank_system')

logger = setup_logging()

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
        
        logger.debug(f"Создана транзакция {transaction_id}: {transaction_type} на сумму {amount:.2f} "
                    f"от {from_account} к {to_account}")
    

    def execute(self) -> bool:
        """Выполнение транзакции"""
        try:
            logger.info(f"Выполнение транзакции {self.transaction_id}")
            self.status = "completed"
            logger.debug(f"Транзакция {self.transaction_id} успешно выполнена")
            return True
        except Exception as e:
            logger.error(f"Ошибка выполнения транзакции {self.transaction_id}: {e}")
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
        
        logger.info(f"Создан банковский счет {account_number} для владельца {account_holder} "
                   f"с начальным балансом {initial_balance:.2f}")
    

    def deposit(self, amount: float, description: str = "") -> bool:
        """Пополнение счета"""
        logger.info(f"Пополнение счета {self.account_number} на сумму {amount:.2f}")
        
        if amount <= 0:
            logger.warning(f"Попытка пополнения счета {self.account_number} неверной суммой: {amount:.2f}")
            print("Сумма пополнения должна быть положительной")
            return False
        
        if not self.is_active:
            logger.warning(f"Попытка пополнения неактивного счета {self.account_number}")
            print("Счет не активен")
            return False
        
        transaction_id = f"TXN{random.randint(100000, 999999)}"
        transaction = Transaction(transaction_id, "EXTERNAL", self.account_number, 
                                 amount, "deposit", description)
        
        if transaction.execute():
            self.balance += amount
            self.transactions.append(transaction)
            logger.info(f"Успешное пополнение счета {self.account_number} на {amount:.2f}. "
                       f"Новый баланс: {self.balance:.2f}")
            print(f"Успешное пополнение: ${amount:.2f}")
            return True
        
        logger.error(f"Ошибка при пополнении счета {self.account_number}")
        return False
    

    def withdraw(self, amount: float, description: str = "") -> bool:
        """Снятие средств"""
        logger.info(f"Снятие средств со счета {self.account_number} суммы {amount:.2f}")
        
        if amount <= 0:
            logger.warning(f"Попытка снятия средств со счета {self.account_number} неверной суммы: {amount:.2f}")
            print("Сумма снятия должна быть положительной")
            return False
        
        if amount > self.balance:
            logger.warning(f"Недостаточно средств на счете {self.account_number}.")
            print("Недостаточно средств на счете")
            return False
        
        if not self.is_active:
            logger.warning(f"Попытка снятия средств с неактивного счета {self.account_number}")
            print("Счет неактивен")
            return False
        
        transaction_id = f"TXN{random.randint(100000, 999999)}"
        transaction = Transaction(transaction_id, self.account_number, "EXTERNAL",
                                 amount, "withdraw", description)
        
        if transaction.execute():
            self.balance -= amount
            self.transactions.append(transaction)
            logger.info(f"Успешное снятие средств со счета {self.account_number} на {amount:.2f}. "
                       f"Новый баланс: {self.balance:.2f}")
            print(f"Успешное снятие: ${amount:.2f}")
            return True
        
        logger.error(f"Ошибка при выполнении транзакции снятия для счета {self.account_number}")
        return False
    

    def transfer(self, to_account: 'BankAccount', amount: float, description: str = "") -> bool:
        """Перевод средств на другой счет"""
        logger.info(f"Перевод {amount:.2f} с {self.account_number} на {to_account.account_number}")
        
        if amount <= 0:
            logger.warning(f"Попытка перевода неверной суммы: {amount:.2f}")
            print("Сумма перевода должна быть положительной")
            return False
        
        if amount > self.balance:
            logger.warning(f"Недостаточно средств для перевода с {self.account_number}. "
                          f"Запрошено: {amount:.2f}, доступно: {self.balance:.2f}")
            print("Недостаточно средств для перевода")
            return False
        
        if not self.is_active or not to_account.is_active:
            logger.warning(f"Попытка перевода с/на неактивный счет. "
                          f"Источник активен: {self.is_active}, Цель активен: {to_account.is_active}")
            print("Один из счетов неактивен")
            return False
        
        transaction_id = f"TXN{random.randint(100000, 999999)}"
        transaction = Transaction(transaction_id, self.account_number, to_account.account_number, 
                                 amount, "transfer", description)
        
        if transaction.execute():
            self.balance -= amount
            to_account.balance += amount
            self.transactions.append(transaction)
            to_account.transactions.append(transaction)
            logger.info(f"Успешный перевод {amount:.2f} с {self.account_number} на {to_account.account_number}. "
                       f"Баланс источника: {self.balance:.2f}, баланс цели: {to_account.balance:.2f}")
            print(f"Успешный перевод: ${amount:.2f} на счет {to_account.account_number}")
            return True
        
        logger.error(f"Ошибка при выполнении перевода между {self.account_number} и {to_account.account_number}")
        return False
    

    def get_balance(self) -> float:
        """Получение текущего баланса"""
        logger.debug(f"Баланса для счета {self.account_number}: {self.balance:.2f}")
        return self.balance
    

    def get_transaction_history(self) -> List[Transaction]:
        """Получение истории транзакций"""
        logger.debug(f"Истории транзакций для счета {self.account_number}. "
                    f"Количество транзакций: {len(self.transactions)}")
        return self.transactions
    

    def get_account_info(self) -> Dict:
        """Получение информации о счете"""
        info = {
            'account_number': self.account_number,
            'account_holder': self.account_holder,
            'balance': self.balance,
            'account_type': self.account_type,
            'is_active': self.is_active,
            'created_date': self.created_date.isoformat(),
            'total_transactions': len(self.transactions)
        }
        
        logger.debug(f"Информация о счете {self.account_number}")
        return info
    

    def deactivate(self) -> bool:
        """Деактивация счета"""
        logger.info(f"Деактивация счета {self.account_number}")
        
        if self.balance == 0:
            self.is_active = False
            logger.info(f"Счет {self.account_number} успешно деактивирован")
            return True
        
        logger.warning(f"Попытка деактивации счета {self.account_number} с ненулевым балансом: {self.balance:.2f}")
        print("Нельзя деактивировать счет с ненулевым балансом")
        return False
    

    def __str__(self) -> str:
        status = "активен" if self.is_active else "неактивен"
        return (f"Счет {self.account_number} ({self.account_holder}): "
                f"${self.balance:.2f}, {status}")


class Bank:
    """Класс для представления банка"""
    
    def __init__(self, bank_name: str):
        self.bank_name = bank_name
        self.accounts: Dict[str, BankAccount] = {}
        self.total_transactions = 0
        
        logger.info(f"Наименование банка: {bank_name}")
    

    def create_account(self, account_holder: str, initial_deposit: float = 0.0) -> BankAccount:
        """Создание нового счета"""
        logger.info(f"Создание счета для {account_holder} с начальным депозитом {initial_deposit:.2f}")
        
        account_number = f"ACC{random.randint(10000000, 99999999)}"
        
        # Проверка на уникальность номера счета
        attempts = 0
        while account_number in self.accounts:
            account_number = f"ACC{random.randint(10000000, 99999999)}"
            attempts += 1
            if attempts > 10:
                logger.error("Превышено количество попыток генерации уникального номера счета")
                raise ValueError("Не удалось сгенерировать уникальный номер счета")
        
        new_account = BankAccount(account_number, account_holder, initial_deposit)
        self.accounts[account_number] = new_account
        
        if initial_deposit > 0:
            new_account.deposit(initial_deposit, "Первоначальный взнос")
        
        logger.info(f"Создан новый счет {account_number} для {account_holder}")
        print(f"Создан новый счет {account_number} для {account_holder}")
        return new_account
    

    def get_account(self, account_number: str) -> Optional[BankAccount]:
        """Получение счета по номеру"""
        account = self.accounts.get(account_number)
        if account:
            logger.debug(f"Найден счет {account_number}")
        else:
            logger.warning(f"Счет {account_number} не найден")
        return account
    

    def close_account(self, account_number: str) -> bool:
        """Закрытие счета"""
        logger.info(f"Закрытие счета {account_number}")
        
        account = self.get_account(account_number)
        if account:
            if account.deactivate():
                del self.accounts[account_number]
                logger.info(f"Счет {account_number} успешно закрыт и удален")
                print(f"Счет {account_number} закрыт")
                return True
        else:
            logger.error(f"Попытка закрытия несуществующего счета {account_number}")
            print(f"Счет {account_number} не найден")
        return False
    

    def get_total_deposits(self) -> float:
        """Общая сумма депозитов в банке"""
        total = sum(account.balance for account in self.accounts.values())
        logger.debug(f"Общая сумма депозитов в банке {self.bank_name}: {total:.2f}")
        return total
    

    def get_bank_statistics(self) -> Dict:
        """Статистика банка"""
        total_accounts = len(self.accounts)
        active_accounts = sum(1 for account in self.accounts.values() if account.is_active)
        total_transactions = sum(len(account.transactions) for account in self.accounts.values())
        
        stats = {
            'bank_name': self.bank_name,
            'total_accounts': total_accounts,
            'active_accounts': active_accounts,
            'total_deposits': self.get_total_deposits(),
            'total_transactions': total_transactions
        }
        
        logger.info(f"Статистика банка {self.bank_name}: {stats}")
        return stats
    

    def find_accounts_by_holder(self, account_holder: str) -> List[BankAccount]:
        """Поиск счетов по владельцу"""
        accounts = [account for account in self.accounts.values() 
                   if account.account_holder.lower() == account_holder.lower()]
        
        logger.debug(f"Поиск счетов для владельца {account_holder}. Найдено: {len(accounts)}")
        return accounts
    

    def save_to_file(self, filename: str) -> bool:
        """Сохранение данных банка в файл"""
        logger.info(f"Сохранение данных банка в файл {filename}")
        
        try:
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
                
            logger.info(f"Данные банка успешно сохранены в {filename}")
            print(f"Данные банка сохранены в {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных в файл {filename}: {e}")
            return False


def DemoBankSystem():
    """Демонстрация работы банковской системы"""
    
    try:
        # Создание банка
        bank = Bank("Банк")
        
        # Создание счетов
        account1 = bank.create_account("Иван Петров", 1000.0)
        account2 = bank.create_account("Мария Сидорова", 500.0)
        account3 = bank.create_account("Алексей Иванов", 200.0)
        
        # Операции со счетами
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
        for account in [account1, account2, account3]:
            info = account.get_account_info()
            print(f"\nВладелец: {info['account_holder']}")
            print(f"Номер счета: {info['account_number']}")
            print(f"Баланс: ${info['balance']:.2f}")
            print(f"Кол-во транзакций: {info['total_transactions']}")
            print(f"Статус: {'активен' if info['is_active'] else 'не активен'}")
        
        # Статистика банка
        stats = bank.get_bank_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: ${value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # История транзакций для первого счета
        for i, transaction in enumerate(account1.get_transaction_history(), 1):
            trans_dict = transaction.to_dict()
            print(f"{i}. {trans_dict['timestamp'][:16]} | "
                  f"{trans_dict['type'].upper():<10} | "
                  f"${trans_dict['amount']:>8.2f} | "
                  f"{trans_dict['description']}")
        
        # Сохранение данных
        bank.save_to_file("bank_data.json")
        
        logger.info("Демонстрация банковской системы успешно завершена")
        print("\nДемонстрация завершена!")
        
    except Exception as e:
        logger.critical(f"Критическая ошибка в демонстрационной системе: {e}")
        print(f"Произошла критическая ошибка: {e}")


if __name__ == "__main__":
    DemoBankSystem()
