"""Restaurant billing system."""

import time

user_names = {}  # username: real name of the user
passwords = {}  # username: password
menu = {}  # dish name: price
dish_numbers = {}  # dish number: dish name
order = {}  # dish name: quantity for the current order
next_dish_number = 1  # number for the next added dish


def register(call):  # call is 1 if called by admin(), 2 if called by customer()
    print("---------------------------Register-------------------------")
    name = input("Enter Name :")
    username = input("Enter Username :")
    password = input("Enter Password :")
    for saved_username in passwords.keys():
        if saved_username == username:
            print("\nUser Already Exits! Please Login")
            if call == 1:
                return login(1)
            else:
                return login(2)
    else:
        passwords[username] = password
        user_names[username] = name
        print("\nHeyy", name, "You Are Registered successfully...\nLogin to Continue")
        if call == 1:
            return login(1)
        else:
            return login(2)


def login(call):
    print("---------------------------Login-------------------------")
    username = input("Enter Username :")
    password = input("Enter Password :")
    user_exists = False
    for saved_username in passwords.keys():
        if saved_username == username:
            user_exists = True
            if passwords[username] == password:
                print("\nLogin Successful")
                print("Welcome ", user_names[username], " Explore things easily here!\n")
                return True
            else:
                print("\nIncorrect Username/Password.... Please Try again")
                if call == 1:
                    return login(1)
                else:
                    return login(2)
    if user_exists == False:
        print("\nUser not found... Please register or login with different account to continue")
        if call == 1:
            return admin()
        else:
            return customer()


def admin():
    print("1.Login")
    print("2.Register")
    auth_choice = int(input("Enter your choice :"))
    if auth_choice == 1:
        logged_in = login(1)
    else:
        logged_in = register(1)

    while logged_in == True:
        print("------------------------------------------------------------")
        print("1.Modify Menu")
        print("2.Display Menu")
        print("3.Logout")
        menu_choice = int(input("Enter your choice :"))
        if menu_choice == 1:
            print("1.Add item")
            print("2.Delete item")
            print("3.Go Back")
            edit_choice = int(input("Enter your choice :"))
            if edit_choice == 1:
                add_item()
            elif edit_choice == 2:
                delete_item()
            else:
                continue
        elif menu_choice == 2:
            display_menu()
        else:
            logged_in = False
            print("Logout Successfully")
            main()  # after logout another user can login, so main() is called here


def customer():
    print("1.Login")
    print("2.Register")
    auth_choice = int(input("Enter your choice :"))
    if auth_choice == 1:
        logged_in = login(2)
    else:
        logged_in = register(2)

    while logged_in == True:
        print("------------------------------------------------------------")
        print("1.Display Menu")
        print("2.Order food")
        print("3.Logout")
        menu_choice = int(input("Enter your choice :"))
        if menu_choice == 1:
            display_menu()
        elif menu_choice == 2:
            logged_in = order_food()
        else:
            logged_in = False
            print("Logout Successfully")
            main()  # after logout another user can login, so main() is called here


def add_item():
    print("---------------------------Add Item-------------------------")
    dish_name = input("Enter Name of Dish :")
    price = int(input("Enter Price :"))
    menu[dish_name] = price
    global next_dish_number
    dish_numbers[next_dish_number] = dish_name
    next_dish_number += 1
    print("\nItem Added Successfully")


def delete_item():
    print("---------------------------Delete Item-------------------------")
    if len(menu) == 0:
        print("Nothing left to delete")
    else:
        dish_name = input("Enter Name of Dish :")
        for i in range(len(menu)):
            if dish_name in menu.keys():
                del menu[dish_name]
                print("\nItem deleted Successfully")
            else:
                print("\nIncorrect Details... try again")
                break


def display_menu():
    print("---------------------------FOOD MENU-------------------------")
    if len(menu) == 0:
        print("Nothing to Show for now")
    else:
        number = 1
        for dish_name in menu.keys():
            print(number, ".", dish_name, "\t= ", menu[dish_name])
            number += 1


def order_food():
    display_menu()
    print("----------------Order Food from above Menu--------------")
    if len(menu) == 0:
        print("Oops...Nothing to order for now!")
        ordering = False
    else:
        ordering = True
    while ordering == True:
        dish_number = int(input("Enter dish number :-"))
        quantity = int(input("Enter Quantity :-"))
        order[dish_numbers[dish_number]] = quantity
        more = input("Want to add more ?(y/n): ")
        if more == "Y" or more == "y":
            ordering = True
        else:
            ordering = False
    print("\nFood Ordered successfuly\n\n")
    return generate_bill()


def animate_bill():
    for i in range(4):
        print("\rGenerating your BILL    ", end="")
        time.sleep(0.5)
        print("\rGenerating your BILL.   ", end="")
        time.sleep(0.5)
        print("\rGenerating your BILL..  ", end="")
        time.sleep(0.5)
        print("\rGenerating your BILL... ", end="")
        time.sleep(0.5)
        print("\rGenerating your BILL....", end="")
        time.sleep(0.5)
    print("\n\tBill Generated Successfully")


def generate_bill():
    animate_bill()
    print("\n-------------------------Bill-------------------------")
    number = 1
    total = 0
    for dish_name in order.keys():
        item_total = order[dish_name] * menu[dish_name]
        print(number, ".", dish_name, "\t", order[dish_name], "x", menu[dish_name], " = ", item_total)
        total += item_total
        number += 1
    print("--------------------------------------------------")
    print(" TOTAL BILL    =      ", total)
    print(" Gst 5%        =      " + "{:.2f}".format((total / 100) * 5))
    print(" Cgst 5%       =      " + "{:.2f}".format((total / 100) * 5))
    total += total / 10
    print("                     --------------")
    print("Grand Total    =      ", total)
    print("                     --------------")
    thank_msg()
    return True  # this will be accepted by order_food() function


def thank_msg():
    print(
        "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\
                     THANK YOU                                         \n\
                    Visit Again !                                        \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    )


def main():
    print("---------------------------Welcome-------------------------")
    print("1.Admin panel")
    print("2.Customer panel")
    print("3.Exit")
    choice = int(input("Enter your choice :"))
    if choice == 1:
        admin()
    elif choice == 2:
        customer()
    else:
        return 0


if __name__ == "__main__":
    main()
