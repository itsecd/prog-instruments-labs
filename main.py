from database import init_database
from auth import register, login
from student_operations import add_student, view_students, search_student, update_student, delete_student


def display_menu():
    print("\n1: Add New Student")
    print("2: View Students")
    print("3: Search Student")
    print("4: Update Student")
    print("5: Delete Student")
    print("6: Exit")


def exit_program():
    print("---------------------------------")
    print(" Thank you for using our system!")
    print("    Created by Siddharth Jain")
    print("       Have a nice day :)")
    print("---------------------------------")


def start(mydb, mycursor):
    print("-------------------------------------")
    print(" Welcome to Student Management System")
    print("-------------------------------------")
    print("1: Login")
    print("2: Register")
    choice = input("Enter your choice: ")
    if choice == "1":
        if login(mycursor):
            return
        else:
            start(mydb, mycursor)
    elif choice == "2":
        if register(mycursor, mydb):
            return
        else:
            start(mydb, mycursor)


def main():
    mydb, mycursor = init_database()
    start(mydb, mycursor)

    while True:
        display_menu()
        choice = input("Enter your choice: ")

        if choice == "1":
            add_student(mycursor, mydb)
        elif choice == "2":
            view_students(mycursor)
        elif choice == "3":
            search_student(mycursor)
        elif choice == "4":
            update_student(mycursor, mydb)
        elif choice == "5":
            delete_student(mycursor, mydb)
        elif choice == "6":
            exit_program()
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()