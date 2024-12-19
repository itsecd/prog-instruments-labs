from model_objects import Customer, ShoppingList, CustomerType, Address


class CustomerMatches:
    def __init__(self):
        self.matchTerm = None
        self.customer = None
        self.duplicates = []

    def has_duplicates(self):
        return self.duplicates

    def add_duplicate(self, duplicate):
        self.duplicates.append(duplicate)


class CustomerDataAccess:
    def __init__(self, db):
        self.customerDataLayer = CustomerDataLayer(db)

    def load_company_customer(self, external_id, company_number):
        matches = CustomerMatches()
        match_by_external_id: Customer = self.customerDataLayer.find_by_external_id(external_id)
        if match_by_external_id is not None:
            matches.customer = match_by_external_id
            matches.matchTerm = "ExternalId"
            match_by_master_id: Customer = self.customerDataLayer.find_by_master_external_id(external_id)
            if match_by_master_id is not None:
                matches.add_duplicate(match_by_master_id)
        else:
            match_by_company_number: Customer = self.customerDataLayer.find_by_company_number(company_number)
            if match_by_company_number is not None:
                matches.customer = match_by_company_number
                matches.matchTerm = "CompanyNumber"

        return matches

    def load_person_customer(self, external_id):
        matches = CustomerMatches()
        match_by_personal_number: Customer = self.customerDataLayer.find_by_external_id(external_id)
        matches.customer = match_by_personal_number
        if match_by_personal_number is not None:
            matches.matchTerm = "ExternalId"
        return matches

    def update_customer_record(self, customer):
        self.customerDataLayer.update_customer_record(customer)

    def create_customer_record(self, customer):
        return self.customerDataLayer.create_customer_record(customer)

    def update_shopping_list(self, customer: Customer, shopping_list: ShoppingList):
        customer.add_shopping_list(shopping_list)
        self.customerDataLayer.update_shopping_list(shopping_list)
        self.customerDataLayer.update_customer_record(customer)


class CustomerDataLayer:
    def __init__(self, conn):
        self.conn = conn
        self.cursor = self.conn.cursor()

    def find_by_external_id(self, external_id):
        self.cursor.execute(
            'SELECT internalId, externalId, masterExternalId, name, customerType,'
            ' companyNumber FROM customers WHERE externalId=?',
            (external_id,))
        customer = self.customer_from_sql_select_fields(self.cursor.fetchone())
        return customer

    def find_address_id(self, customer):
        self.cursor.execute('SELECT addressId FROM customers WHERE internalId=?', (customer.internalId,))
        (addressId,) = self.cursor.fetchone()
        if addressId:
            return int(addressId)
        return None

    def customer_from_sql_select_fields(self, fields):
        if not fields:
            return None

        customer = Customer(internal_id=fields[0], external_id=fields[1], master_external_id=fields[2], name=fields[3],
                            customer_type=CustomerType(fields[4]), company_number=fields[5])
        address_id = self.find_address_id(customer)
        if address_id:
            self.cursor.execute('SELECT street, city, postalCode FROM addresses WHERE addressId=?',
                                (address_id,))
            addresses = self.cursor.fetchone()
            if addresses:
                (street, city, postalCode) = addresses
                address = Address(street, city, postalCode)
                customer.address = address
        self.cursor.execute('SELECT shoppinglistId FROM customer_shoppinglists WHERE customerId=?',
                            (customer.internalId,))
        shopping_lists = self.cursor.fetchall()
        for sl in shopping_lists:
            self.cursor.execute('SELECT products FROM shoppinglists WHERE shoppinglistId=?', (sl[0],))
            products_as_str = self.cursor.fetchone()
            products = products_as_str[0].split(", ")
            customer.add_shopping_list(ShoppingList(products))
        return customer

    def find_by_master_external_id(self, master_external_id):
        self.cursor.execute(
            'SELECT internalId, externalId, masterExternalId, name, customerType,'
            ' companyNumber FROM customers WHERE masterExternalId=?',
            (master_external_id,))
        return self.customer_from_sql_select_fields(self.cursor.fetchone())

    def find_by_company_number(self, company_number):
        self.cursor.execute(
            'SELECT internalId, externalId, masterExternalId, name, customerType,'
            ' companyNumber FROM customers WHERE companyNumber=?',
            (company_number,))
        return self.customer_from_sql_select_fields(self.cursor.fetchone())

    def create_customer_record(self, customer):
        customer.internalId = self.next_id("customers")
        self.cursor.execute('INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?);', (
            customer.internalId, customer.externalId, customer.masterExternalId, customer.name,
            customer.customerType.value,
            customer.companyNumber, None))
        if customer.address:
            address_id = self.next_id("addresses")
            self.cursor.execute('INSERT INTO addresses VALUES (?, ?, ?, ?)', (
                address_id, customer.address.street, customer.address.city, customer.address.postalCode))
            self.cursor.execute('UPDATE customers set addressId=? WHERE internalId=?',
                                (address_id, customer.internalId))

        if customer.shoppingLists:
            for sl in customer.shoppingLists:
                data = ", ".join(sl)
                self.cursor.execute('SELECT shoppinglistId FROM shoppinglists WHERE products=?', (data,))
                shopping_list_id = self.cursor.fetchone()
                if not shopping_list_id:
                    shopping_list_id = self.next_id("shoppinglists")
                    self.cursor.execute('INSERT INTO shoppinglists VALUES (?, ?)', (shopping_list_id, data))
                self.cursor.execute('INSERT INTO customer_shoppinglists VALUES (?, ?)',
                                    (customer.internalId, shopping_list_id))
        self.conn.commit()
        return customer

    def next_id(self, table_name):
        self.cursor.execute(f'SELECT MAX(ROWID) AS max_id FROM {table_name};')
        (old_id,) = self.cursor.fetchone()
        if old_id:
            return int(old_id) + 1
        else:
            return 1

    def update_customer_record(self, customer):
        self.cursor.execute(
            'Update customers set externalId=?, masterExternalId=?, name=?,'
            ' customerType=?, companyNumber=? WHERE internalId=?',
            (customer.externalId, customer.masterExternalId, customer.name, customer.customerType.value,
             customer.companyNumber, customer.internalId))
        if customer.address:
            address_id = self.find_address_id(customer)
            if not address_id:
                address_id = self.next_id("addresses")
                self.cursor.execute('INSERT INTO addresses VALUES (?, ?, ?, ?)', (
                    address_id, customer.address.street, customer.address.city, customer.address.postalCode))
                self.cursor.execute('UPDATE customers set addressId=? WHERE internalId=?',
                                    (address_id, customer.internalId))

        self.cursor.execute('DELETE FROM customer_shoppinglists WHERE customerId=?', (customer.internalId,))
        if customer.shoppingLists:
            for sl in customer.shoppingLists:
                products = ", ".join(sl.products)
                self.cursor.execute('SELECT shoppinglistId FROM shoppinglists WHERE products=?', (products,))
                shopping_list_ids = self.cursor.fetchone()
                if shopping_list_ids is not None:
                    (shoppinglistId,) = shopping_list_ids
                    self.cursor.execute('INSERT INTO customer_shoppinglists VALUES (?, ?)',
                                        (customer.internalId, shoppinglistId))
                else:
                    shoppinglist_id = self.next_id("shoppinglists")
                    self.cursor.execute('INSERT INTO shoppinglists VALUES (?, ?)', (shoppinglist_id, products))
                    self.cursor.execute('INSERT INTO customer_shoppinglists VALUES (?, ?)',
                                        (customer.internalId, shoppinglist_id))

        self.conn.commit()

    def update_shopping_list(self, shopping_list):
        pass