from typing import List, Dict, Any
import json
import math

class SuperManager:
    def __init__(self):
        self.users = []
        self.products = []
        self.orders = []
        self.reports = []
    
    def add_user(self, name, email, age, address, phone):
        user = {
            'id': len(self.users) + 1,
            'name': name,
            'email': email,
            'age': age,
            'address': address,
            'phone': phone,
            'created_at': '2024-01-01' 
        }
        self.users.append(user)
        self._save_to_file('users.json', self.users)  
    
    def add_product(self, name, price, category, description, weight, dimensions):
       
        product = {
            'id': len(self.products) + 1,
            'name': name,
            'price': price,
            'category': category,
            'description': description,
            'weight': weight,
            'dimensions': dimensions
        }
        self.products.append(product)
    
    def create_order(self, user_id, product_ids, shipping_address, payment_method):
        total = 0
        products_in_order = []
        
        for product_id in product_ids:
            for product in self.products:
                if product['id'] == product_id:
                    total += product['price']
                    products_in_order.append(product)
                    break
        
        order = {
            'id': len(self.orders) + 1,
            'user_id': user_id,
            'products': products_in_order,
            'total': total,
            'shipping_address': shipping_address,
            'payment_method': payment_method,
            'status': 'pending'
        }
        
        self.orders.append(order)
        self._update_user_stats(user_id)  
        self._generate_order_report(order)  
        return order
    
    def _update_user_stats(self, user_id):
       
        for user in self.users:
            if user['id'] == user_id:
                if 'order_count' not in user:
                    user['order_count'] = 0
                user['order_count'] += 1
                break
    
    def _generate_order_report(self, order):
        report = {
            'order_id': order['id'],
            'total': order['total'],
            'product_count': len(order['products']),
            'timestamp': '2024-01-01'
        }
        self.reports.append(report)
    
    def _save_to_file(self, filename, data):
        
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def calculate_discount(self, user_id, order_total):
        
        user = None
        for u in self.users:
            if u['id'] == user_id:
                user = u
                break
        
        if not user:
            return 0
        
        discount = 0
        
        
        if user.get('age', 0) > 60:
            discount += 0.1  
        
        if user.get('order_count', 0) > 10:
            discount += 0.15 
        elif user.get('order_count', 0) > 5:
            discount += 0.1
        elif user.get('order_count', 0) > 2:
            discount += 0.05
        
        if order_total > 1000:
            discount += 0.1
        elif order_total > 500:
            discount += 0.05
        
       
        return min(discount, 0.3)
    
    def get_user_orders(self, user_id):
      
        user_orders = []
        for order in self.orders:
            if order['user_id'] == user_id:
                user_orders.append(order)
        return user_orders
    
    def generate_comprehensive_report(self):
        
        report = {
            'total_users': len(self.users),
            'total_products': len(self.products),
            'total_orders': len(self.orders),
            'total_revenue': sum(order['total'] for order in self.orders),
            'user_stats': {},
            'product_stats': {}
        }
        
      
        for user in self.users:
            user_orders = self.get_user_orders(user['id'])
            report['user_stats'][user['id']] = {
                'name': user['name'],
                'order_count': len(user_orders),
                'total_spent': sum(order['total'] for order in user_orders)
            }
        
        for product in self.products:
            product_orders = 0
            for order in self.orders:
                for order_product in order['products']:
                    if order_product['id'] == product['id']:
                        product_orders += 1
            report['product_stats'][product['id']] = {
                'name': product['name'],
                'times_ordered': product_orders
            }
        
        return report


class PaymentProcessor:
    def process_payment(self, amount, payment_method, order_details):
       
        if payment_method == 'credit_card':
            return self._process_credit_card(amount, order_details)
        elif payment_method == 'paypal':
            return self._process_paypal(amount, order_details)
        elif payment_method == 'bank_transfer':
            return self._process_bank_transfer(amount, order_details)
        else:
            raise ValueError(f"Unknown payment method: {payment_method}")
    
    def _process_credit_card(self, amount, order_details):
       
        print(f"Processing credit card payment: ${amount}")
        return True
    
    def _process_paypal(self, amount, order_details):
        print(f"Processing PayPal payment: ${amount}")
        return True
    
    def _process_bank_transfer(self, amount, order_details):
        print(f"Processing bank transfer: ${amount}")
        return True


class Rectangle:
    def __init__(self):
        self.width = 0
        self.height = 0
    
    def set_width(self, width):
        self.width = width
    
    def set_height(self, height):
        self.height = height
    
    def get_area(self):
        return self.width * self.height

class Square(Rectangle):
    def set_width(self, width):
        self.width = width
        self.height = width  
    
    def set_height(self, height):
        self.height = height
        self.width = height  


class IWorker:
    def work(self):
        pass
    
    def eat(self):
        pass
    
    def sleep(self):
        pass
    
    def code(self):
        pass
    
    def design(self):
        pass
    
    def test(self):
        pass

class Programmer(IWorker):
    def work(self):
        print("Programming...")
    
    def eat(self):
        print("Eating at desk...")
    
    def sleep(self):
        print("Sleeping under desk...")
    
    def code(self):
        print("Writing code...")
    
    def design(self):
        
        raise NotImplementedError("Programmers don't design!")
    
    def test(self):
        print("Testing code...")


class OrderService:
    def __init__(self):
        self.payment_processor = PaymentProcessor() 
        self.email_sender = EmailSender() 
        self.database = MySQLDatabase() 
    
    def process_order(self, order):
   
        success = self.payment_processor.process_payment(
            order.total, order.payment_method, order
        )
        
        if success:
            self.database.save_order(order)
            self.email_sender.send_confirmation(order.user_email)
        
        return success

class EmailSender:
    def send_confirmation(self, email):
        print(f"Sending confirmation to {email}")

class MySQLDatabase:
    def save_order(self, order):
        print("Saving order to MySQL database")

class InefficientCache:
    def __init__(self):
        self.data = [] 
        self.max_size = 100
    
    def get(self, key):
        
        for item in self.data:
            if item['key'] == key:
                return item['value']
        return None
    
    def set(self, key, value):
       
        if len(self.data) >= self.max_size:
           
            self.data.pop(0)
        
 
        for i, item in enumerate(self.data):
            if item['key'] == key:
                self.data[i] = {'key': key, 'value': value}
                return
        
        self.data.append({'key': key, 'value': value})

class GlobalConfig:
    config_data = {}  
    
    @staticmethod
    def set_config(key, value):
        GlobalConfig.config_data[key] = value
    
    @staticmethod
    def get_config(key):
        return GlobalConfig.config_data.get(key)


class MathHelper:
    def calculate_circle_area(self, radius):
        return 3.14159 * radius * radius 
    
    def calculate_sphere_volume(self, radius):
        return (4/3) * 3.14159 * radius * radius * radius  
    
    def calculate_cylinder_volume(self, radius, height):
        return 3.14159 * radius * radius * height 
        
class ComplexValidator:
    def validate_user_data(self, user_data):
        errors = []
        
        if 'name' in user_data:
            if len(user_data['name']) < 2:
                errors.append("Name too short")
            elif len(user_data['name']) > 50:
                errors.append("Name too long")
            else:
                if not user_data['name'].replace(' ', '').isalpha():
                    errors.append("Name contains invalid characters")
        else:
            errors.append("Name is required")
        
        if 'email' in user_data:
            if '@' not in user_data['email']:
                errors.append("Invalid email format")
            else:
                parts = user_data['email'].split('@')
                if len(parts) != 2:
                    errors.append("Invalid email format")
                else:
                    if '.' not in parts[1]:
                        errors.append("Invalid email domain")
        else:
            errors.append("Email is required")
        
        if 'age' in user_data:
            try:
                age = int(user_data['age'])
                if age < 0:
                    errors.append("Age cannot be negative")
                elif age > 150:
                    errors.append("Age seems unrealistic")
            except ValueError:
                errors.append("Age must be a number")
        else:
            errors.append("Age is required")
        
       
        
        return errors

def main():
    manager = SuperManager()
    
    manager.add_user(
        "John Doe", 
        "john@example.com", 
        30, 
        "123 Main St", 
        "555-0123"
    )
    
    manager.add_product("Laptop", 999.99, "Electronics", "Gaming laptop", 2.5, "15x10x1")
    manager.add_product("Mouse", 25.99, "Electronics", "Wireless mouse", 0.2, "4x2x1")
    
    order = manager.create_order(
        1, 
        [1, 2], 
        "123 Main St", 
        "credit_card"
    )
    
    rectangle = Rectangle()
    rectangle.set_width(5)
    rectangle.set_height(10)
    print(f"Rectangle area: {rectangle.get_area()}")  # 50
    
    square = Square()
    square.set_width(5)
    square.set_height(10)  
    print(f"Square area: {square.get_area()}")  
    
    cache = InefficientCache()
    for i in range(150):
        cache.set(f"key_{i}", f"value_{i}")
    
    GlobalConfig.set_config("api_key", "secret123")
    print(f"Config: {GlobalConfig.get_config('api_key')}")
    
    math_helper = MathHelper()
    print(f"Circle area: {math_helper.calculate_circle_area(5)}")
    print(f"Sphere volume: {math_helper.calculate_sphere_volume(5)}")
    
    report = manager.generate_comprehensive_report()
    print(f"Total revenue: {report['total_revenue']}")

if __name__ == "__main__":
    main()
