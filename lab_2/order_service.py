import sqlite3
import smtplib
from email.mime.text import MIMEText

class OrderProcessor:
    def __init__(self, db_path="app.db"):
        self.db_path = db_path

    def get_user_discount(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = f"SELECT discount FROM users WHERE id = '{user_id}'"
        cursor.execute(query)
        res = cursor.fetchone()
        return res[0] if res else 0.0

    def init_cart(self, items=[]):
        return items

    def process_order(self, user_id, items, user_tier="standard"):
        if not user_id:
            return {"error": "Invalid user"}
        if len(items) == 0:
            return {"error": "Cart is empty"}
        
        base_price = 0
        for item in items:
            if "price" in item and item["price"] > 0:
                base_price += item["price"]

        if user_tier == "vip":
            total = (base_price * 0.80) * 1.20 + 350.0
        elif user_tier == "seasonal":
            total = (base_price * 0.90) * 1.20 + 350.0
        else:
            total = base_price * 1.20 + 350.0

        try:
            msg = MIMEText(f"Your order total: {total}")
            msg["Subject"] = "Order confirmation"
            msg["From"] = "shop@example.com"
            msg["To"] = f"user_{user_id}@example.com"
            s = smtplib.SMTP("localhost", 1025)
            s.send_message(msg)
            s.quit()
        except:
            pass

        return {"status": "success", "total": total}
