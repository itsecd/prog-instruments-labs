from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from dataclasses_json import dataclass_json


class CustomerType(Enum):
    PERSON = 1
    COMPANY = 2


@dataclass_json
@dataclass
class ShoppingList:
    products: List[str] = field(default_factory=list)


@dataclass_json
@dataclass(frozen=True)
class Address:
    street: str
    city: str
    postalCode: str


@dataclass_json
@dataclass(frozen=True)
class ExternalCustomer:
    externalId: str
    name: str
    isCompany: bool
    companyNumber: Optional[str]
    preferredStore: str
    postalAddress: Address
    shoppingLists: List[ShoppingList] = field(default_factory=list)


class Customer:
    def __init__(self, internal_id: str = None, external_id: str = None, master_external_id: str = None,
                 name: str = None, customer_type: CustomerType = None, company_number: str = None):
        self.internalId = internal_id
        self.externalId = external_id
        self.masterExternalId = master_external_id
        self.name = name
        self.customerType = customer_type
        self.companyNumber = company_number
        self.shoppingLists = []
        self.address = None

    def add_shopping_list(self, shopping_list):
        self.shoppingLists.append(shopping_list)