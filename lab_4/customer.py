from abc import ABC, abstractmethod
from typing import List, Union
import util


class BaseCustomer(ABC):
    """Abstract base class for customers."""

    def __init__(self, name: str):
        self.name = name
        self.first_name = ''
        self.last_name = ''
        self._parse_name(name)

    def __str__(self):
        return f'<Customer {self.name}>'

    def _parse_name(self, name: str):
        """Parse full name into first and last name."""
        name_parts = name.split(' ')
        if len(name_parts) > 1:
            self.first_name = util.get_first_name(name)
            self.last_name = ' '.join(name_parts[1:])
        else:
            self.first_name = name
            self.last_name = name

    @abstractmethod
    def switch_company(self, new_company: str):
        pass


class Customer(BaseCustomer):
    """Concrete customer implementation."""

    def __init__(self, name: str, email: Union[str, List[str]], company: str):
        super().__init__(name)
        self.email = email
        self.company = company

    def add_email(self, new_email: str):
        """Add new email address to customer."""
        if isinstance(self.email, str):
            self.email = [self.email, new_email]
        else:
            self.email.append(new_email)

    def switch_company(self, new_company: str):
        """Switch customer to new company."""
        self.company = new_company


