import datetime
import json
import os

TAX_RATE = 0.2
DISCOUNT_THRESHOLD = 1000
LARGE_ORDER_DISCOUNT = 0.9
VIP_DISCOUNT = 0.95
ELECTRONICS_DISCOUNT = 0.95
CLOTHES_DISCOUNT = 0.9
SMALL_ORDER_THRESHOLD = 500
FREE_DELIVERY_THRESHOLD = 5000
DELIVERY_FEE_SMALL = 50
DELIVERY_FEE_MEDIUM = 30
USERS_DB_FILE = "users.json"

class OrderSystem:
    def __init__(self):
        self.users = {}
        self.orders = []
        self.load_users()

    def load_users(self):
        if os.path.exists(USERS_DB_FILE):
            with open(USERS_DB_FILE, 'r') as f:
                self.users = json.load(f)
        else:
            self.users = {
                "user1": {"name": "Ivan", "email": "ivan@mail.ru", "is_vip": False},
                "user2": {"name": "Petr", "email": "petr@mail.ru", "is_vip": True}
            }

    def _find_order_by_id(self, order_id):
        for order in self.orders:
            if order['id'] == order_id:
                return order
        return None

    def _validate_user(self, user_id):
        if user_id not in self.users:
            print("Ошибка: Пользователь не найден")
            return False
        return True

    def _validate_items(self, items):
        for item in items:
            if item['price'] <= 0:
                print(f"Ошибка: Неверная цена для {item['name']}")
                return False
            if item['quantity'] <= 0:
                print(f"Ошибка: Неверное количество для {item['name']}")
                return False
        return True

    def _calculate_items_total(self, items):
        total = 0
        for item in items:
            if item['category'] == 'electronics':
                total += item['price'] * item['quantity'] * ELECTRONICS_DISCOUNT
            elif item['category'] == 'clothes':
                total += item['price'] * item['quantity'] * CLOTHES_DISCOUNT
            else:
                total += item['price'] * item['quantity']
        return total

    def _apply_discounts(self, total, user):
        if total > DISCOUNT_THRESHOLD:
            total = total * LARGE_ORDER_DISCOUNT
            print("Применена скидка 10% за крупную сумму")
        if user['is_vip']:
            total = total * VIP_DISCOUNT
            print("Применена VIP скидка 5%")
        return total

    def _apply_tax_and_delivery(self, total):
        total = total + (total * TAX_RATE)
        if total < SMALL_ORDER_THRESHOLD:
            total = total + DELIVERY_FEE_SMALL
            print(f"Добавлена стоимость доставки: {DELIVERY_FEE_SMALL}")
        elif total > FREE_DELIVERY_THRESHOLD:
            print("Доставка бесплатная")
        else:
            total = total + DELIVERY_FEE_MEDIUM
            print(f"Добавлена стоимость доставки: {DELIVERY_FEE_MEDIUM}")
        return total

    def _save_order(self, user_id, items, total):
        order_id = len(self.orders) + 1
        order_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        order = {
            "id": order_id,
            "user_id": user_id,
            "items": items,
            "total": round(total, 2),
            "date": order_date,
            "status": "created"
        }
        self.orders.append(order)
        return order

    def process_order(self, user_id, items):
        print(f"Начинаем обработку заказа для {user_id}")
        
        if not self._validate_user(user_id):
            return False
        if not self._validate_items(items):
            return False
            
        user = self.users[user_id]
        
        total = self._calculate_items_total(items)
        total = self._apply_discounts(total, user)
        total = self._apply_tax_and_delivery(total)

        order = self._save_order(user_id, items, total)
        
        print(f"Заказ {order['id']} успешно создан. Итоговая сумма: {order['total']}")
        return order

    def print_receipt(self, order_id):
        print(f"--- ЧЕК ПО ЗАКАЗУ #{order_id} ---")
        order = self._find_order_by_id(order_id)
        
        if not order:
            print("Заказ не найден")
            return

        user = self.users[order['user_id']]
        print(f"Покупатель: {user['name']} ({user['email']})")
        print(f"Дата: {order['date']}")
        print("Товары:")
        for item in order['items']:
            print(f" - {item['name']} x {item['quantity']} = {item['price'] * item['quantity']}")
        print(f"ИТОГО К ОПЛАТЕ: {order['total']}")
        print("-------------------------")

    def change_order_status(self, order_id, new_status):
        valid_statuses = ["created", "paid", "shipped", "delivered", "cancelled"]
        if new_status not in valid_statuses:
            print("Ошибка: Неверный статус")
            return False
            
        order = self._find_order_by_id(order_id)
        
        if order:
            if order['status'] == "cancelled" and new_status != "cancelled":
                print("Ошибка: Нельзя изменить отмененный заказ")
                return False
            if order['status'] == "delivered" and new_status != "delivered":
                print("Ошибка: Нельзя изменить доставленный заказ")
                return False
            order['status'] = new_status
            print(f"Статус заказа {order_id} изменен на {new_status}")
            return True
            
        print("Заказ не найден")
        return False

    def generate_sales_report(self):
        print("=== ОТЧЕТ ПО ПРОДАЖАМ ===")
        total_revenue = 0
        cancelled_count = 0
        for order in self.orders:
            if order['status'] == 'cancelled':
                cancelled_count += 1
                continue
            total_revenue += order['total']
        
        print(f"Всего заказов: {len(self.orders)}")
        print(f"Отмененных заказов: {cancelled_count}")
        print(f"Общая выручка: {round(total_revenue, 2)}")
        
        if len(self.orders) - cancelled_count > 0:
            avg = total_revenue / (len(self.orders) - cancelled_count)
            print(f"Средний чек: {round(avg, 2)}")
        else:
            print("Средний чек: 0")
        print("=========================")

    def get_user_orders(self, user_id, status_filter=None):
        if user_id not in self.users:
            print("Пользователь не найден")
            return []
        
        result = []
        for order in self.orders:
            if order['user_id'] == user_id:
                if status_filter:
                    if order['status'] == status_filter:
                        result.append(order)
                else:
                    result.append(order)
        return result


if __name__ == "__main__":
    system = OrderSystem()
    
    test_items = [
        {"name": "Ноутбук", "price": 1000, "quantity": 1, "category": "electronics"},
        {"name": "Футболка", "price": 50, "quantity": 2, "category": "clothes"},
        {"name": "Книга", "price": 20, "quantity": 1, "category": "books"}
    ]
    
    order = system.process_order("user1", test_items)
    
    if order:
        system.print_receipt(order['id'])
        system.change_order_status(order['id'], "paid")
        system.generate_sales_report()
        user_orders = system.get_user_orders("user1", "paid")
        print(f"Найдено оплаченных заказов: {len(user_orders)}")