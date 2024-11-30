import sqlite3


class Database:
    def __init__(self, data):
        self.con = sqlite3.connect(data)
        self.cur = self.con.cursor()
        sql = """
        CREATE TABLE IF NOT EXISTS employees(
            id Integer Primary Key,
            name text,
            age text,
            doj text,
            email text,
            gender text,
            contact text,
            address text
        )
        """
        self.cur.execute(sql)
        self.con.commit()

    # Insert Function
    def insert(self, name, age, doj, email, gender, contact, address):
        self.cur.execute("insert into employees values (NULL,?,?,?,?,?,?,?)",
                         (name, age, doj, email, gender, contact, address))
        self.con.commit()

    # Fetch All Data from DB
    def fetch(self):
        self.cur.execute("SELECT * from employees")
        rows = self.cur.fetchall()
        # print(rows)
        return rows

    # Delete a Record in DB
    def remove(self, idx):
        self.cur.execute("delete from employees where id=?", (idx,))
        self.con.commit()

    # Update a Record in DB
    def update(self, idx, name, age, doj, email, gender, contact, address):
        self.cur.execute(
            "update employees set name=?, age=?, doj=?, email=?, gender=?, contact=?, address=? where id=?",
            (name, age, doj, email, gender, contact, address, idx))
        self.con.commit()

    # Clear all a Record in DB
    def clear_ALL(self):
        for i in range(len(self.fetch())):
            self.remove(i+1)
        self.con.commit()

    def get_data_ind(self, ind):
        rows = self.fetch()
        return rows[ind-1]

    def find_data(self, name):
        self.cur.execute("SELECT * FROM employees WHERE name LIKE ?", ('%' + name + '%',))
        rows = self.cur.fetchall()
        return bool(rows)

    def data_is_null(self):
        if len(self.fetch()) == 0:
            return True
        return False