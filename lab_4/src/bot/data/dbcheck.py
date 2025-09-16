import sqlite3
import json

conn = sqlite3.connect('data/users.db')
cursor = conn.cursor()

def print_all_users():
    cursor.execute('SELECT * FROM users')
    all_users = cursor.fetchall()
    
    for user in all_users:
        userid, tasks_json = user
        tasks = json.loads(tasks_json) 
        print(f'UserID: {userid}, Tasks: {tasks}')
        
print_all_users()
conn.close()
input("end.")