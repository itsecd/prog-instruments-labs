"""Restaurant billing system."""

import time

GST_PERCENT = 5
CGST_PERCENT = 5
ANIMATION_REPEATS = 4  # how many times the dots are repeated
ANIMATION_DELAY = 0.5  # seconds between animation frames


def read_int(prompt, minimum=None):
    """Ask for a whole number until the user enters a correct one."""
    while True:
        text = input(prompt)
        try:
            value = int(text)
        except ValueError:
            print("Please enter a number")
            continue
        if minimum is not None and value < minimum:
            print("The number must be at least", minimum)
            continue
        return value


def register(users):
    print("---------------------------Register-------------------------")
    name = input("Enter Name :")
    username = input("Enter Username :")
    password = input("Enter Password :")
    if username in users:
        print("\nUser Already Exits! Please Login")
    else:
        users[username] = {"name": name, "password": password}
        print("\nHeyy", name, "You Are Registered successfully...\nLogin to Continue")
    return login(users)


def login(users):
    while True:
        print("---------------------------Login-------------------------")
        username = input("Enter Username :")
        password = input("Enter Password :")
        if username not in users:
            print("\nUser not found... Please register or login with different account to continue")
            return False
        if users[username]["password"] == password:
            print("\nLogin Successful")
            print("Welcome ", users[username]["name"], " Explore things easily here!\n")
            return True
        print("\nIncorrect Username/Password.... Please Try again")


def authenticate(users):
    """Ask to login or register until the user is logged in."""
    while True:
        print("1.Login")
        print("2.Register")
        auth_choice = read_int("Enter your choice :")
        if auth_choice == 1:
            logged_in = login(users)
        else:
            logged_in = register(users)
        if logged_in:
            return


def admin(users, menu):
    authenticate(users)
    while True:
        print("------------------------------------------------------------")
        print("1.Modify Menu")
        print("2.Display Menu")
        print("3.Logout")
        menu_choice = read_int("Enter your choice :")
        if menu_choice == 1:
            print("1.Add item")
            print("2.Delete item")
            print("3.Go Back")
            edit_choice = read_int("Enter your choice :")
            if edit_choice == 1:
                add_item(menu)
            elif edit_choice == 2:
                delete_item(menu)
        elif menu_choice == 2:
            display_menu(menu)
        else:
            print("Logout Successfully")
            return


def customer(users, menu):
    authenticate(users)
    while True:
        print("------------------------------------------------------------")
        print("1.Display Menu")
        print("2.Order food")
        print("3.Logout")
        menu_choice = read_int("Enter your choice :")
        if menu_choice == 1:
            display_menu(menu)
        elif menu_choice == 2:
            order_food(menu)
        else:
            print("Logout Successfully")
            return


def add_item(menu):
    print("---------------------------Add Item-------------------------")
    dish_name = input("Enter Name of Dish :").strip()
    if dish_name == "":
        print("\nName of the dish can't be empty")
        return
    price = read_int("Enter Price :", 1)
    menu[dish_name] = price
    print("\nItem Added Successfully")


def delete_item(menu):
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


def display_menu(menu):
    print("---------------------------FOOD MENU-------------------------")
    if len(menu) == 0:
        print("Nothing to Show for now")
    else:
        for number, (dish_name, price) in enumerate(menu.items(), start=1):
            print(number, ".", dish_name, "\t= ", price)


def order_food(menu):
    display_menu(menu)
    print("----------------Order Food from above Menu--------------")
    if len(menu) == 0:
        print("Oops...Nothing to order for now!")
        return
    dishes = list(menu)  # dish number in the menu = position in this list + 1
    order = {}  # dish name: quantity
    while True:
        dish_number = read_int("Enter dish number :-")
        if not 1 <= dish_number <= len(dishes):
            print("There is no dish with this number, try again")
            continue
        quantity = read_int("Enter Quantity :-", 1)
        order[dishes[dish_number - 1]] = quantity
        more = input("Want to add more ?(y/n): ")
        if more.lower() != "y":
            break
    print("\nFood Ordered successfuly\n\n")
    generate_bill(order, menu)


def animate_bill():
    frames = ["    ", ".   ", "..  ", "... ", "...."]
    for _ in range(ANIMATION_REPEATS):
        for frame in frames:
            print("\rGenerating your BILL" + frame, end="")
            time.sleep(ANIMATION_DELAY)
    print("\n\tBill Generated Successfully")


def generate_bill(order, menu):
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
    users = {}  # username: {"name": ..., "password": ...}
    menu = {}  # dish name: price
    while True:
        print("---------------------------Welcome-------------------------")
        print("1.Admin panel")
        print("2.Customer panel")
        print("3.Exit")
        choice = read_int("Enter your choice :")
        if choice == 1:
            admin(users, menu)
        elif choice == 2:
            customer(users, menu)
        else:
            return


if __name__ == "__main__":
    main()
