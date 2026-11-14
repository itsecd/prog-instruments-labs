"""Restaurant billing system."""

import time

GST_PERCENT = 5
CGST_PERCENT = 5
ANIMATION_REPEATS = 4  # how many times the dots are repeated
ANIMATION_DELAY = 0.5  # seconds between animation frames

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
    if username in passwords:
        print("\nUser Already Exits! Please Login")
    else:
        passwords[username] = password
        user_names[username] = name
        print("\nHeyy", name, "You Are Registered successfully...\nLogin to Continue")
    return login(call)


def login(call):
    print("---------------------------Login-------------------------")
    username = input("Enter Username :")
    password = input("Enter Password :")
    if username not in passwords:
        print("\nUser not found... Please register or login with different account to continue")
        if call == 1:
            return admin()
        else:
            return customer()
    if passwords[username] == password:
        print("\nLogin Successful")
        print("Welcome ", user_names[username], " Explore things easily here!\n")
        return True
    print("\nIncorrect Username/Password.... Please Try again")
    return login(call)


def admin():
    print("1.Login")
    print("2.Register")
    auth_choice = int(input("Enter your choice :"))
    if auth_choice == 1:
        logged_in = login(1)
    else:
        logged_in = register(1)

    while logged_in:
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

    while logged_in:
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
        if dish_name in menu:
            del menu[dish_name]
            print("\nItem deleted Successfully")
        else:
            print("\nIncorrect Details... try again")


def display_menu():
    print("---------------------------FOOD MENU-------------------------")
    if len(menu) == 0:
        print("Nothing to Show for now")
    else:
        for number, (dish_name, price) in enumerate(menu.items(), start=1):
            print(number, ".", dish_name, "\t= ", price)


def order_food():
    display_menu()
    print("----------------Order Food from above Menu--------------")
    if len(menu) == 0:
        print("Oops...Nothing to order for now!")
        return True
    while True:
        dish_number = int(input("Enter dish number :-"))
        quantity = int(input("Enter Quantity :-"))
        order[dish_numbers[dish_number]] = quantity
        more = input("Want to add more ?(y/n): ")
        if more.lower() != "y":
            break
    print("\nFood Ordered successfuly\n\n")
    return generate_bill()


def animate_bill():
    frames = ["    ", ".   ", "..  ", "... ", "...."]
    for _ in range(ANIMATION_REPEATS):
        for frame in frames:
            print("\rGenerating your BILL" + frame, end="")
            time.sleep(ANIMATION_DELAY)
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
    gst = total * GST_PERCENT / 100
    cgst = total * CGST_PERCENT / 100
    grand_total = total + gst + cgst
    print(" TOTAL BILL    =      ", total)
    print(" Gst " + str(GST_PERCENT) + "%        =      " + "{:.2f}".format(gst))
    print(" Cgst " + str(CGST_PERCENT) + "%       =      " + "{:.2f}".format(cgst))
    print("                     --------------")
    print("Grand Total    =      ", grand_total)
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
"""Restaurant billing system."""

import time

GST_PERCENT = 5
CGST_PERCENT = 5
ANIMATION_REPEATS = 4  # how many times the dots are repeated
ANIMATION_DELAY = 0.5  # seconds between animation frames

user_names = {}  # username: real name of the user
passwords = {}  # username: password
menu = {}  # dish name: price
dish_numbers = {}  # dish number: dish name
order = {}  # dish name: quantity for the current order
next_dish_number = 1  # number for the next added dish


def register():
    print("---------------------------Register-------------------------")
    name = input("Enter Name :")
    username = input("Enter Username :")
    password = input("Enter Password :")
    if username in passwords:
        print("\nUser Already Exits! Please Login")
    else:
        passwords[username] = password
        user_names[username] = name
        print("\nHeyy", name, "You Are Registered successfully...\nLogin to Continue")
    return login()


def login():
    while True:
        print("---------------------------Login-------------------------")
        username = input("Enter Username :")
        password = input("Enter Password :")
        if username not in passwords:
            print("\nUser not found... Please register or login with different account to continue")
            return False
        if passwords[username] == password:
            print("\nLogin Successful")
            print("Welcome ", user_names[username], " Explore things easily here!\n")
            return True
        print("\nIncorrect Username/Password.... Please Try again")


def authenticate():
    """Ask to login or register until the user is logged in."""
    while True:
        print("1.Login")
        print("2.Register")
        auth_choice = int(input("Enter your choice :"))
        if auth_choice == 1:
            logged_in = login()
        else:
            logged_in = register()
        if logged_in:
            return


def admin():
    authenticate()
    while True:
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
        elif menu_choice == 2:
            display_menu()
        else:
            print("Logout Successfully")
            return


def customer():
    authenticate()
    while True:
        print("------------------------------------------------------------")
        print("1.Display Menu")
        print("2.Order food")
        print("3.Logout")
        menu_choice = int(input("Enter your choice :"))
        if menu_choice == 1:
            display_menu()
        elif menu_choice == 2:
            order_food()
        else:
            print("Logout Successfully")
            return


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
        if dish_name in menu:
            del menu[dish_name]
            print("\nItem deleted Successfully")
        else:
            print("\nIncorrect Details... try again")


def display_menu():
    print("---------------------------FOOD MENU-------------------------")
    if len(menu) == 0:
        print("Nothing to Show for now")
    else:
        for number, (dish_name, price) in enumerate(menu.items(), start=1):
            print(number, ".", dish_name, "\t= ", price)


def order_food():
    display_menu()
    print("----------------Order Food from above Menu--------------")
    if len(menu) == 0:
        print("Oops...Nothing to order for now!")
        return
    while True:
        dish_number = int(input("Enter dish number :-"))
        quantity = int(input("Enter Quantity :-"))
        order[dish_numbers[dish_number]] = quantity
        more = input("Want to add more ?(y/n): ")
        if more.lower() != "y":
            break
    print("\nFood Ordered successfuly\n\n")
    generate_bill()


def animate_bill():
    frames = ["    ", ".   ", "..  ", "... ", "...."]
    for _ in range(ANIMATION_REPEATS):
        for frame in frames:
            print("\rGenerating your BILL" + frame, end="")
            time.sleep(ANIMATION_DELAY)
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
    gst = total * GST_PERCENT / 100
    cgst = total * CGST_PERCENT / 100
    grand_total = total + gst + cgst
    print(" TOTAL BILL    =      ", total)
    print(" Gst " + str(GST_PERCENT) + "%        =      " + "{:.2f}".format(gst))
    print(" Cgst " + str(CGST_PERCENT) + "%       =      " + "{:.2f}".format(cgst))
    print("                     --------------")
    print("Grand Total    =      ", grand_total)
    print("                     --------------")
    thank_msg()


def thank_msg():
    print(
        "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\
                     THANK YOU                                         \n\
                    Visit Again !                                        \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    )


def main():
    while True:
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
            return


if __name__ == "__main__":
    main()
