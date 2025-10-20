import hashlib


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register(mycursor, mydb):
    print("-------------------------")
    print("--------Register---------")
    print("-------------------------")
    id = input("Enter your Id: ")
    password = input("Enter your Password: ")
    hashed_password = hash_password(password)

    mycursor.execute("insert into admin values (%s, %s)",   (id, hashed_password))
    mydb.commit()
    print("Registered successfully!")
    input("Press Enter to continue!")
    return login(mycursor)


def login(mycursor):
    print("-------------")
    print("--- login ---")
    print("-------------")
    id = input("Enter the id: ")
    password = input("Enter the password: ")
    hashed_password = hash_password(password)

    mycursor.execute(
        "select * from admin where id = %s and password = %s", (id, hashed_password)
    )
    result = mycursor.fetchall()

    if len(result) == 1:
        print("-------------------------------------")
        print(f"--Welcome {id}, what you want to do!--")
        print("-------------------------------------")
        return True
    else:
        return False