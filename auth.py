def register(mycursor, mydb):
    print("-------------------------")
    print("--------Register---------")
    print("-------------------------")
    id = input("Enter your Id: ")
    password = input("Enter your Password: ")

    mycursor.execute("insert into admin values('" + id + "','" + password + "')")
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

    mycursor.execute(
        "select * from admin where id=('" + id + "') and password=('" + password + "')"
    )
    result = mycursor.fetchall()

    if len(result) == 1:
        print("-------------------------------------")
        print(f"--Welcome {id}, what you want to do!--")
        print("-------------------------------------")
        return True
    else:
        return False