# Система аутентификации пользователей

import json
import os

users_file = "users.json"

def load_users():
    global users_file
    f = open(users_file, "r")
    data = f.read()
    users = json.loads(data)
    f.close()
    return users

def save_users(users):
    global users_file
    f = open(users_file, "w")
    f.write(json.dumps(users))
    f.close()

def register_user(username, password, email):
    users = load_users()
    
    for u in users:
        if u["username"] == username:
            return False
    
    user = {}
    user["username"] = username
    user["password"] = password
    user["email"] = email
    user["role"] = "user"
    
    users.append(user)
    save_users(users)
    return True

def login(username, password):
    users = load_users()
    
    for u in users:
        if u["username"] == username and u["password"] == password:
            return True
    
    return False

def change_password(username, old_password, new_password):
    users = load_users()
    
    for u in users:
        if u["username"] == username:
            if u["password"] == old_password:
                u["password"] = new_password
                save_users(users)
                return True
            else:
                return False
    
    return False

def get_user_info(username):
    users = load_users()
    
    for u in users:
        if u["username"] == username:
            return {
                "username": u["username"],
                "email": u["email"],
                "role": u["role"]
            }
    
    return None

def delete_user(username, password):
    global users_file
    users = load_users()
    
    for i in range(len(users)):
        if users[i]["username"] == username:
            if users[i]["password"] == password:
                users.pop(i)
                save_users(users)
                return True
            else:
                return False
    
    return False

def main():
    print("1. Register")
    print("2. Login")
    print("3. Change password")
    print("4. Get user info")
    print("5. Delete account")
    
    choice = input("Choice: ")
    
    if choice == "1":
        username = input("Username: ")
        password = input("Password: ")
        email = input("Email: ")
        if register_user(username, password, email):
            print("Registered!")
        else:
            print("User exists!")
    
    elif choice == "2":
        username = input("Username: ")
        password = input("Password: ")
        if login(username, password):
            print("Logged in!")
        else:
            print("Invalid credentials!")
    
    elif choice == "3":
        username = input("Username: ")
        old_pass = input("Old password: ")
        new_pass = input("New password: ")
        if change_password(username, old_pass, new_pass):
            print("Password changed!")
        else:
            print("Failed!")
    
    elif choice == "4":
        username = input("Username: ")
        info = get_user_info(username)
        if info:
            print(info)
        else:
            print("User not found!")
    
    elif choice == "5":
        username = input("Username: ")
        password = input("Password: ")
        if delete_user(username, password):
            print("Account deleted!")
        else:
            print("Failed!")

if __name__ == "__main__":
    main()