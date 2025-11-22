import json
import datetime
import os
from typing import List, Dict, Optional, Any
from enum import Enum


class OrderStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class OrderType(Enum):
    DINE_IN = "dine_in"
    TAKEAWAY = "takeaway"


class MenuItemType(Enum):
    MAIN = "main"
    SALAD = "salad"
    DRINK = "drink"
    DESSERT = "dessert"


class UserType(Enum):
    CUSTOMER = "customer"
    ADMIN = "admin"
    STAFF = "staff"


class MenuItem:
    def __init__(self, id: int, name: str, price: float, item_type: MenuItemType, available: bool):
        self.id = id
        self.name = name
        self.price = price
        self.type = item_type
        self.available = available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "type": self.type.value,
            "available": self.available
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MenuItem':
        return cls(
            id=data["id"],
            name=data["name"],
            price=data["price"],
            type=MenuItemType(data["type"]),
            available=data["available"]
        )


class User:
    def __init__(self, id: int, username: str, password: str, email: str, user_type: UserType, created_at: str):
        self.id = id
        self.username = username
        self.password = password
        self.email = email
        self.user_type = user_type
        self.created_at = created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "password": self.password,
            "email": self.email,
            "user_type": self.user_type.value,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        return cls(
            id=data["id"],
            username=data["username"],
            password=data["password"],
            email=data["email"],
            user_type=UserType(data["user_type"]),
            created_at=data["created_at"]
        )


class Order:
    def __init__(self, order_id: int, customer_name: str, items: List[MenuItem], order_type: OrderType,
                 table_number: Optional[int] = None):
        self.id = order_id
        self.customer_name = customer_name
        self.items = items
        self.order_type = order_type
        self.table_number = table_number
        self.status = OrderStatus.PENDING
        self.total = self._calculate_total()
        self.timestamp = datetime.datetime.now().isoformat()

    def _calculate_total(self) -> float:
        total = 0
        for item in self.items:
            total += item.price

        # Apply discounts
        if total > 1000:
            total = total * 0.9  # 10% discount
        elif total > 500:
            total = total * 0.95  # 5% discount

        # Add service charge for dine-in
        if self.order_type == OrderType.DINE_IN:
            total = total * 1.1  # 10% service charge

        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_name": self.customer_name,
            "items": [item.to_dict() for item in self.items],
            "order_type": self.order_type.value,
            "table_number": self.table_number,
            "status": self.status.value,
            "total": self.total,
            "timestamp": self.timestamp
        }


class DataRepository:
    def __init__(self):
        self.orders_data = []
        self.menu_items = []
        self.users = []

    def load_from_file(self, filename: str, factory_class):
        filepath = filename
        if not os.path.exists(filepath):
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [factory_class.from_dict(item) for item in data]

    def save_to_file(self, filename: str, data):
        with open(filename, "w", encoding="utf-8") as f:
            json_data = [item.to_dict() for item in data]
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    def load_all_data(self):
        try:
            self.orders_data = self.load_from_file("orders.json", Order)
        except:
            self.orders_data = []

        try:
            self.menu_items = self.load_from_file("menu.json", MenuItem)
            if not self.menu_items:
                # Default menu
                self.menu_items = [
                    MenuItem(1, "Пицца Маргарита", 450, MenuItemType.MAIN, True),
                    MenuItem(2, "Паста Карбонара", 380, MenuItemType.MAIN, True),
                    MenuItem(3, "Цезарь", 280, MenuItemType.SALAD, True),
                    MenuItem(4, "Кока-Кола", 120, MenuItemType.DRINK, True),
                    MenuItem(5, "Тирамису", 220, MenuItemType.DESSERT, False)
                ]
        except:
            self.menu_items = []

        try:
            self.users = self.load_from_file("users.json", User)
        except:
            self.users = []

    def save_all_data(self):
        self.save_to_file("orders.json", self.orders_data)
        self.save_to_file("menu.json", self.menu_items)
        self.save_to_file("users.json", self.users)

    def find_menu_item_by_id(self, item_id):
        for menu_item in self.menu_items:
            if menu_item.id == item_id:
                return menu_item
        return None

    def find_order_by_id(self, order_id):
        for order in self.orders_data:
            if order.id == order_id:
                return order
        return None


class OrderService:
    def __init__(self, repository):
        self.repository = repository

    def create_order(self, customer_name, items, table_number, order_type):
        # Validate items
        valid_items = []
        for item_id in items:
            menu_item = self.repository.find_menu_item_by_id(item_id)
            if menu_item and menu_item.available:
                valid_items.append(menu_item)
            else:
                print(f"Item {item_id} not available or not found")

        if len(valid_items) == 0:
            print("No valid items in order")
            return None

        # Create order
        order_id = len(self.repository.orders_data) + 1
        order = Order(order_id, customer_name, valid_items, OrderType(order_type), table_number)

        self.repository.orders_data.append(order)
        self.repository.save_all_data()
        return order

    def update_order_status(self, order_id, new_status):
        order = self.repository.find_order_by_id(order_id)
        if order:
            order.status = OrderStatus(new_status)
            self.repository.save_all_data()
            return True
        return False

    def process_bulk_orders(self, orders_list):
        results = []
        for order_data in orders_list:
            try:
                order = self.create_order(
                    order_data["customer_name"],
                    order_data["items"],
                    order_data.get("table_number", 0),
                    order_data.get("order_type", OrderType.DINE_IN.value)
                )
                if order:
                    results.append({"success": True, "order_id": order.id})
                else:
                    results.append({"success": False, "error": "Failed to create order"})
            except Exception as e:
                results.append({"success": False, "error": str(e)})
        return results

    def get_orders_by_status(self, status):
        result = []
        for order in self.repository.orders_data:
            if order.status.value == status:
                result.append(order)
        return result


class ReportService:
    def __init__(self, repository):
        self.repository = repository

    def calculate_daily_revenue(self, date_str):
        total_revenue = 0
        completed_orders = 0
        for order in self.repository.orders_data:
            order_date = order.timestamp[:10]
            if order_date == date_str and order.status == OrderStatus.COMPLETED:
                total_revenue += order.total
                completed_orders += 1

        print(f"Date: {date_str}")
        print(f"Completed orders: {completed_orders}")
        print(f"Total revenue: {total_revenue}")
        return total_revenue

    def get_popular_items(self, limit=5):
        item_count = {}
        for order in self.repository.orders_data:
            if order.status == OrderStatus.COMPLETED:
                for item in order.items:
                    item_id = item.id
                    item_count[item_id] = item_count.get(item_id, 0) + 1

        # Convert to list and sort
        popular = []
        for item_id, count in item_count.items():
            for menu_item in self.repository.menu_items:
                if menu_item.id == item_id:
                    popular.append({
                        "item": menu_item,
                        "count": count
                    })
                    break

        popular.sort(key=lambda x: x["count"], reverse=True)
        return popular[:limit]

    def generate_period_report(self, start_date, end_date):
        report = {
            "period": f"{start_date} to {end_date}",
            "total_orders": 0,
            "completed_orders": 0,
            "total_revenue": 0,
            "popular_items": [],
            "orders_by_type": {}
        }

        for order in self.repository.orders_data:
            order_date = order.timestamp[:10]
            if start_date <= order_date <= end_date:
                report["total_orders"] += 1
                if order.status == OrderStatus.COMPLETED:
                    report["completed_orders"] += 1
                    report["total_revenue"] += order.total

                # Count by type
                order_type = order.order_type.value
                report["orders_by_type"][order_type] = report["orders_by_type"].get(order_type, 0) + 1

        # Get popular items for the period
        temp_orders = [o for o in self.repository.orders_data if
                       start_date <= o.timestamp[:10] <= end_date and o.status == OrderStatus.COMPLETED]
        item_count = {}
        for order in temp_orders:
            for item in order.items:
                item_id = item.id
                item_count[item_id] = item_count.get(item_id, 0) + 1

        popular = []
        for item_id, count in item_count.items():
            for menu_item in self.repository.menu_items:
                if menu_item.id == item_id:
                    popular.append({
                        "item": menu_item.name,
                        "count": count,
                        "revenue": count * menu_item.price
                    })
                    break

        popular.sort(key=lambda x: x["count"], reverse=True)
        report["popular_items"] = popular[:5]

        return report


class UserService:
    def __init__(self, repository):
        self.repository = repository

    def register_user(self, username, password, email, user_type="customer"):
        # Check if username exists
        for user in self.repository.users:
            if user.username == username:
                return False

        new_user = User(
            id=len(self.repository.users) + 1,
            username=username,
            password=password,
            email=email,
            user_type=UserType(user_type),
            created_at=datetime.datetime.now().isoformat()
        )

        self.repository.users.append(new_user)
        self.repository.save_all_data()
        return True

    def authenticate_user(self, username, password):
        for user in self.repository.users:
            if user.username == username and user.password == password:
                return user
        return None


class MenuService:
    def __init__(self, repository):
        self.repository = repository

    def add_menu_item(self, name, price, item_type):
        new_id = max([item.id for item in self.repository.menu_items]) + 1 if self.repository.menu_items else 1
        new_item = MenuItem(
            id=new_id,
            name=name,
            price=price,
            item_type=MenuItemType(item_type),
            available=True
        )
        self.repository.menu_items.append(new_item)
        self.repository.save_all_data()
        return new_id

    def update_menu_item(self, item_id, new_price=None, new_availability=None):
        for item in self.repository.menu_items:
            if item.id == item_id:
                if new_price is not None:
                    item.price = new_price
                if new_availability is not None:
                    item.available = new_availability
                self.repository.save_all_data()
                return True
        return False

    def get_available_menu_items(self):
        return [item for item in self.repository.menu_items if item.available]


class RestaurantSystem:
    def __init__(self):
        self.repository = DataRepository()
        self.order_service = OrderService(self.repository)
        self.report_service = ReportService(self.repository)
        self.user_service = UserService(self.repository)
        self.menu_service = MenuService(self.repository)

    def initialize(self):
        self.repository.load_all_data()

    def save_data(self):
        self.repository.save_all_data()


class RestaurantCLI:
    def __init__(self, system: RestaurantSystem):
        self.system = system

    def show_main_menu(self):
        print("\n=== Restaurant Management System ===")
        print("1. Add Order")
        print("2. View Orders")
        print("3. Update Order Status")
        print("4. Manage Menu")
        print("5. Reports")
        print("6. User Management")
        print("7. Exit")

    def _show_available_menu(self):
        print("Available menu items:")
        available_items = self.system.menu_service.get_available_menu_items()
        for item in available_items:
            print(f"{item.id}. {item.name} - {item.price} руб.")

    def handle_add_order(self):
        print("\n--- Add New Order ---")
        customer_name = input("Customer name: ")

        self._show_available_menu()

        items_input = input("Enter item IDs (comma-separated): ")
        items = [int(x.strip()) for x in items_input.split(",")]

        order_type = input("Order type (dine_in/takeaway): ")
        table_number = 0
        if order_type == "dine_in":
            table_number = int(input("Table number: "))

        order = self.system.order_service.create_order(customer_name, items, table_number, order_type)
        if order:
            print(f"Order created successfully! Order ID: {order.id}")
        else:
            print("Failed to create order")

    def handle_view_orders(self):
        status = input("Enter status to filter (or press Enter for all): ")
        if status:
            orders = self.system.order_service.get_orders_by_status(status)
        else:
            orders = self.system.repository.orders_data

        print(f"\n--- Orders ({len(orders)}) ---")
        for order in orders:
            print(f"ID: {order.id}, Customer: {order.customer_name}, "
                  f"Total: {order.total}, Status: {order.status.value}")

    def handle_daily_revenue(self):
        date_str = input("Enter date (YYYY-MM-DD): ")
        self.system.report_service.calculate_daily_revenue(date_str)

    def run(self):
        self.system.initialize()

        while True:
            self.show_main_menu()
            choice = input("Enter your choice: ")

            if choice == "1":
                self.handle_add_order()
            elif choice == "2":
                self.handle_view_orders()
            elif choice == "3":
                order_id = int(input("Order ID: "))
                new_status = input("New status: ")
                if self.system.order_service.update_order_status(order_id, new_status):
                    print("Order status updated!")
                else:
                    print("Order not found")
            elif choice == "4":
                print("Menu management not implemented in this version")
            elif choice == "5":
                self.handle_daily_revenue()
            elif choice == "6":
                print("User management not implemented in this version")
            elif choice == "7":
                print("Goodbye!")
                break
            else:
                print("Invalid choice!")


def main():
    system = RestaurantSystem()
    cli = RestaurantCLI(system)
    cli.run()


if __name__ == "__main__":
    main()