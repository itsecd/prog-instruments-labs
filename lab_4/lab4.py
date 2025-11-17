import json
import datetime
import os

orders_data = []
menu_items = []
users = []


def load_data():
    global orders_data, menu_items, users
    try:
        if os.path.exists("orders.json"):
            with open("orders.json", "r") as f:
                orders_data = json.load(f)
        else:
            orders_data = []
    except:
        orders_data = []

    try:
        if os.path.exists("menu.json"):
            with open("menu.json", "r") as f:
                menu_items = json.load(f)
        else:
            # Default menu
            menu_items = [
                {"id": 1, "name": "Пицца Маргарита", "price": 450, "type": "main", "available": True},
                {"id": 2, "name": "Паста Карбонара", "price": 380, "type": "main", "available": True},
                {"id": 3, "name": "Цезарь", "price": 280, "type": "salad", "available": True},
                {"id": 4, "name": "Кока-Кола", "price": 120, "type": "drink", "available": True},
                {"id": 5, "name": "Тирамису", "price": 220, "type": "dessert", "available": False}
            ]
    except:
        menu_items = []

    try:
        if os.path.exists("users.json"):
            with open("users.json", "r") as f:
                users = json.load(f)
        else:
            users = []
    except:
        users = []


def save_data():
    with open("orders.json", "w") as f:
        json.dump(orders_data, f)
    with open("menu.json", "w") as f:
        json.dump(menu_items, f)
    with open("users.json", "w") as f:
        json.dump(users, f)


def add_order(customer_name, items, table_number, order_type):
    global orders_data

    # Validate items
    valid_items = []
    total = 0
    for item_id in items:
        found = False
        for menu_item in menu_items:
            if menu_item["id"] == item_id and menu_item["available"]:
                valid_items.append(menu_item)
                total += menu_item["price"]
                found = True
                break
        if not found:
            print(f"Item {item_id} not available or not found")

    if len(valid_items) == 0:
        print("No valid items in order")
        return False

    # Apply discounts
    if total > 1000:
        total = total * 0.9  # 10% discount
    elif total > 500:
        total = total * 0.95  # 5% discount

    # Add service charge for dine-in
    if order_type == "dine_in":
        total = total * 1.1  # 10% service charge

    # Create order
    order_id = len(orders_data) + 1
    order = {
        "id": order_id,
        "customer_name": customer_name,
        "items": valid_items,
        "table_number": table_number if order_type == "dine_in" else None,
        "order_type": order_type,
        "total": total,
        "status": "pending",
        "timestamp": datetime.datetime.now().isoformat()
    }

    orders_data.append(order)
    save_data()
    return order_id


def update_order_status(order_id, new_status):
    global orders_data
    for order in orders_data:
        if order["id"] == order_id:
            order["status"] = new_status
            order["updated_at"] = datetime.datetime.now().isoformat()
            save_data()
            return True
    return False


def calculate_daily_revenue(date_str):
    total_revenue = 0
    completed_orders = 0
    for order in orders_data:
        order_date = order["timestamp"][:10]
        if order_date == date_str and order["status"] == "completed":
            total_revenue += order["total"]
            completed_orders += 1

    print(f"Date: {date_str}")
    print(f"Completed orders: {completed_orders}")
    print(f"Total revenue: {total_revenue}")
    return total_revenue


def get_orders_by_status(status):
    result = []
    for order in orders_data:
        if order["status"] == status:
            result.append(order)
    return result


def add_menu_item(name, price, item_type):
    global menu_items
    new_id = max([item["id"] for item in menu_items]) + 1 if menu_items else 1
    new_item = {
        "id": new_id,
        "name": name,
        "price": price,
        "type": item_type,
        "available": True
    }
    menu_items.append(new_item)
    save_data()
    return new_id


def update_menu_item(item_id, new_price=None, new_availability=None):
    global menu_items
    for item in menu_items:
        if item["id"] == item_id:
            if new_price is not None:
                item["price"] = new_price
            if new_availability is not None:
                item["available"] = new_availability
            save_data()
            return True
    return False


def get_popular_items(limit=5):
    item_count = {}
    for order in orders_data:
        if order["status"] == "completed":
            for item in order["items"]:
                item_id = item["id"]
                item_count[item_id] = item_count.get(item_id, 0) + 1

    # Convert to list and sort
    popular = []
    for item_id, count in item_count.items():
        for menu_item in menu_items:
            if menu_item["id"] == item_id:
                popular.append({
                    "item": menu_item,
                    "count": count
                })
                break

    popular.sort(key=lambda x: x["count"], reverse=True)
    return popular[:limit]


def register_user(username, password, email, user_type="customer"):
    global users
    # Check if username exists
    for user in users:
        if user["username"] == username:
            return False

    new_user = {
        "id": len(users) + 1,
        "username": username,
        "password": password,  # Insecure - no hashing!
        "email": email,
        "user_type": user_type,
        "created_at": datetime.datetime.now().isoformat()
    }

    users.append(new_user)
    save_data()
    return True


def authenticate_user(username, password):
    for user in users:
        if user["username"] == username and user["password"] == password:
            return user
    return None


def process_bulk_orders(orders_list):
    results = []
    for order_data in orders_list:
        try:
            order_id = add_order(
                order_data["customer_name"],
                order_data["items"],
                order_data.get("table_number", 0),
                order_data.get("order_type", "dine_in")
            )
            results.append({"success": True, "order_id": order_id})
        except Exception as e:
            results.append({"success": False, "error": str(e)})
    return results


def generate_report(start_date, end_date):
    report = {
        "period": f"{start_date} to {end_date}",
        "total_orders": 0,
        "completed_orders": 0,
        "total_revenue": 0,
        "popular_items": [],
        "orders_by_type": {}
    }

    for order in orders_data:
        order_date = order["timestamp"][:10]
        if start_date <= order_date <= end_date:
            report["total_orders"] += 1
            if order["status"] == "completed":
                report["completed_orders"] += 1
                report["total_revenue"] += order["total"]

            # Count by type
            order_type = order["order_type"]
            report["orders_by_type"][order_type] = report["orders_by_type"].get(order_type, 0) + 1

    # Get popular items for the period
    temp_orders = [o for o in orders_data if
                   start_date <= o["timestamp"][:10] <= end_date and o["status"] == "completed"]
    item_count = {}
    for order in temp_orders:
        for item in order["items"]:
            item_id = item["id"]
            item_count[item_id] = item_count.get(item_id, 0) + 1

    popular = []
    for item_id, count in item_count.items():
        for menu_item in menu_items:
            if menu_item["id"] == item_id:
                popular.append({
                    "item": menu_item["name"],
                    "count": count,
                    "revenue": count * menu_item["price"]
                })
                break

    popular.sort(key=lambda x: x["count"], reverse=True)
    report["popular_items"] = popular[:5]

    return report


# UI functions (mixing logic with presentation)
def show_main_menu():
    print("\n=== Restaurant Management System ===")
    print("1. Add Order")
    print("2. View Orders")
    print("3. Update Order Status")
    print("4. Manage Menu")
    print("5. Reports")
    print("6. User Management")
    print("7. Exit")


def handle_add_order():
    print("\n--- Add New Order ---")
    customer_name = input("Customer name: ")

    # Show menu
    print("Available menu items:")
    for item in menu_items:
        if item["available"]:
            print(f"{item['id']}. {item['name']} - {item['price']} руб.")

    items_input = input("Enter item IDs (comma-separated): ")
    items = [int(x.strip()) for x in items_input.split(",")]

    order_type = input("Order type (dine_in/takeaway): ")
    table_number = 0
    if order_type == "dine_in":
        table_number = int(input("Table number: "))

    order_id = add_order(customer_name, items, table_number, order_type)
    if order_id:
        print(f"Order created successfully! Order ID: {order_id}")
    else:
        print("Failed to create order")


def handle_view_orders():
    status = input("Enter status to filter (or press Enter for all): ")
    if status:
        orders = get_orders_by_status(status)
    else:
        orders = orders_data

    print(f"\n--- Orders ({len(orders)}) ---")
    for order in orders:
        print(f"ID: {order['id']}, Customer: {order['customer_name']}, "
              f"Total: {order['total']}, Status: {order['status']}")


def main():
    load_data()

    while True:
        show_main_menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            handle_add_order()
        elif choice == "2":
            handle_view_orders()
        elif choice == "3":
            order_id = int(input("Order ID: "))
            new_status = input("New status: ")
            if update_order_status(order_id, new_status):
                print("Order status updated!")
            else:
                print("Order not found")
        elif choice == "4":
            # Menu management would go here
            print("Menu management not implemented in this version")
        elif choice == "5":
            date_str = input("Enter date (YYYY-MM-DD): ")
            calculate_daily_revenue(date_str)
        elif choice == "6":
            # User management would go here
            print("User management not implemented in this version")
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()