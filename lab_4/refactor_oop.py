import collections
from abc import ABC, abstractmethod
from typing import List


class Person:
    """Base class representing a person."""

    def __init__(self, company_website: str = '', area_code: str = '+44',
                 number: str = '01212158624'):
        self.company_website = company_website
        Telephone = collections.namedtuple('Telephone', ['area_code', 'number'])
        self.tel = Telephone(area_code=area_code, number=number)

    def get_company_website_extension(self) -> str:
        """Extract website extension."""
        if not self.company_website:
            return ''

        result = self.company_website.replace('/', '').split('.')
        if len(result) > 1:
            return result[-1]
        else:
            print('Not a valid domain')
            return ''

    def dial(self) -> str:
        """Format phone number for dialing."""
        return self.tel.area_code + self.tel.number


class BusinessEntity(Person):
    """Base class for business entities with stage information."""

    def __init__(self, stage: str, **kwargs):
        super().__init__(**kwargs)
        self.stage = stage


class Customer(BusinessEntity):
    def __init__(self, **kwargs):
        super().__init__(stage='customer', **kwargs)


class Lead(BusinessEntity):
    def __init__(self, **kwargs):
        super().__init__(stage='lead', **kwargs)


class Opportunity(BusinessEntity):
    def __init__(self, **kwargs):
        super().__init__(stage='opportunity', **kwargs)


class Employee:
    def notify(self, message: str):
        """Notify employee with message."""
        pass


class PricingStrategy(ABC):
    """Abstract base class for pricing strategies."""

    BASE_PRICE = 100
    TAX_RATE = 0.75

    @abstractmethod
    def calculate_price(self, units: int) -> float:
        pass


class SMBPricingStrategy(PricingStrategy):
    """Pricing strategy for Small/Medium Business."""

    def calculate_price(self, units: int) -> float:
        base_price = self.BASE_PRICE * units * 0.8
        tax = base_price * self.TAX_RATE * 0.8
        return base_price * tax


class EnterprisePricingStrategy(PricingStrategy):
    """Pricing strategy for Enterprise."""

    def calculate_price(self, units: int) -> float:
        base_price = self.BASE_PRICE * units
        tax = base_price * self.TAX_RATE
        return base_price * tax


class Company:
    """Represents a company with employees and pricing."""

    def __init__(self, website: str = '', size: int = 0, industry: str = ''):
        self.website = website
        self.size = size
        self.industry = industry
        self.employees: List[Employee] = []
        self.pricing_strategy: PricingStrategy = None

    def set_pricing_strategy(self, strategy: PricingStrategy):
        """Set pricing strategy for the company."""
        self.pricing_strategy = strategy

    def get_pricing(self, units: int) -> float:
        """Calculate price using current pricing strategy."""
        if not self.pricing_strategy:
            raise ValueError("Pricing strategy not set")
        return self.pricing_strategy.calculate_price(units)

    def get_key_employee(self) -> Employee:
        """Get key employee for notifications."""
        return self.employees[0] if self.employees else None

    def notify_key_employee(self, message: str):
        """Notify key employee directly."""
        key_employee = self.get_key_employee()
        if key_employee:
            key_employee.notify(message)


class SmallMediumBusiness(Company):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_pricing_strategy(SMBPricingStrategy())


class Enterprise(Company):
    def __init__(self, account_executive=None, **kwargs):
        super().__init__(**kwargs)
        self.account_executive = account_executive or {}
        self.set_pricing_strategy(EnterprisePricingStrategy())

    def notify_account_executive(self, email_func, message: str):
        """Notify account executive using email function."""
        if 'email' in self.account_executive:
            email_func(self.account_executive['email'])
