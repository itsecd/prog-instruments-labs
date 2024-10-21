from dataclasses import dataclass, field
from typing import List, Optional

from customer_data_access import CustomerMatches
from model_objects import Customer, CustomerType, ExternalCustomer


class ConflictException(Exception):
    pass


class CustomerSync:

    def __init__(self, customerDataAccess: CustomerDataAccess) -> None:
        self.customerDataAccess = customerDataAccess

    def syncWithDataLayer(self, externalCustomer: ExternalCustomer) -> bool:
        customerMatches: CustomerMatches
        if externalCustomer.isCompany:
            customerMatches = self.loadCompany(externalCustomer)
        else:
            customerMatches = self.loadPerson(externalCustomer)

        customer = customerMatches.customer

        if customer is None:
            customer = Customer()
            customer.externalId = externalCustomer.externalId
            customer.masterExternalId = externalCustomer.externalId

        self.populateFields(externalCustomer, customer)

        created = False
        if customer.internalId is None:
            customer = self.createCustomer(customer)
            created = True
        else:
            self.updateCustomer(customer)

        self.updateContactInfo(externalCustomer, customer)

        if customerMatches.has_duplicates:
            for duplicate in customerMatches.duplicates:
                self.updateDuplicate(externalCustomer, duplicate)

        self.updateRelations(externalCustomer, customer)
        self.updatePreferredStore(externalCustomer, customer)

        return created

    def updateRelations(self, externalCustomer: ExternalCustomer, customer: Customer) -> None:
        consumerShoppingLists = externalCustomer.shoppingLists
        for consumerShoppingList in consumerShoppingLists:
            self.customerDataAccess.updateShoppingList(
                customer, consumerShoppingList
            )

    def updateCustomer(self, customer: Customer) -> Customer:
        return self.customerDataAccess.updateCustomerRecord(customer)

    def updateDuplicate(self, externalCustomer: ExternalCustomer, 
                        duplicate: Optional[Customer]) -> None:
        if duplicate is None:
            duplicate = Customer()
            duplicate.externalId = externalCustomer.externalId
            duplicate.masterExternalId = externalCustomer.externalId

        duplicate.name = externalCustomer.name

        if duplicate.internalId is None:
            self.createCustomer(duplicate)
        else:
            self.updateCustomer(duplicate)

    def updatePreferredStore(self, externalCustomer: ExternalCustomer, 
                              customer: Customer) -> None:
        customer.preferredStore = externalCustomer.preferredStore

    def createCustomer(self, customer: Customer) -> Customer:
        return self.customerDataAccess.createCustomerRecord(customer)

    def populateFields(self, externalCustomer: ExternalCustomer, 
                       customer: Customer) -> None:
        customer.name = externalCustomer.name
        if externalCustomer.isCompany:
            customer.companyNumber = externalCustomer.companyNumber
            customer.customerType = CustomerType.COMPANY
        else:
            customer.customerType = CustomerType.PERSON

    def updateContactInfo(self, externalCustomer: ExternalCustomer, 
                      customer: Customer) -> None:
        customer.address = externalCustomer.postalAddress

    def loadCompany(self, externalCustomer: ExternalCustomer) -> CustomerMatches:
        externalId = externalCustomer.externalId
        companyNumber = externalCustomer.companyNumber

        customerMatches = self.customerDataAccess.loadCompanyCustomer(
            externalId, companyNumber
        )

        if (customerMatches.customer is not None and 
                CustomerType.COMPANY != customerMatches.customer.customerType):
            raise ConflictException(
                f"Existing customer for externalCustomer {externalId} "
                "already exists and is not a company"
            )

        if "ExternalId" == customerMatches.matchTerm:
            customerCompanyNumber = customerMatches.customer.companyNumber
            if companyNumber != customerCompanyNumber:
                customerMatches.customer.masterExternalId = None
                customerMatches.add_duplicate(customerMatches.customer)
                customerMatches.customer = None
                customerMatches.matchTerm = None

        elif "CompanyNumber" == customerMatches.matchTerm:
            customerExternalId = customerMatches.customer.externalId
            if (customerExternalId is not None and 
                    externalId != customerExternalId):
                raise ConflictException(
                    f"Existing customer for externalCustomer {companyNumber} "
                    f"doesn't match external id {externalId} instead found "
                    f"{customerExternalId}"
                )

            customer = customerMatches.customer
            customer.externalId = externalId
            customer.masterExternalId = externalId
            customerMatches.addDuplicate(None)

        return customerMatches

    def loadPerson(self, externalCustomer: ExternalCustomer) -> CustomerMatches:
        externalId = externalCustomer.externalId

        customerMatches = self.customerDataAccess.loadPersonCustomer(externalId)

        if customerMatches.customer is not None:
            if CustomerType.PERSON != customerMatches.customer.customerType:
                raise ConflictException(
                    f"Existing customer for externalCustomer {externalId} "
                    "already exists and is not a person"
                )

            if "ExternalId" != customerMatches.matchTerm:
                customer = customerMatches.customer
                customer.externalId = externalId
                customer.masterExternalId = externalId

        return customerMatches

