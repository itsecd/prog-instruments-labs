import pickle
import os
from random import randint


def head(data):
    print
    print
    "=" * 167
    print
    string = "# |" + " " * 8 + data + " " * 8 + "| #"
    print
    string.center(167, "-")
    print
    print
    "=" * 167
    print


def menu(data):
    print
    ml = []
    for i in data:
        ml.append(len(i))
    max_len = max(ml)
    for i in data:
        s = i + " " * (max_len - len(i))
        print
        s.center(167)
    print


def output(data):
    print
    string = ">" + " " * 5 + data + " " * 5 + "<"
    print
    string.center(167, "-")
    print


def maxlenl(text):
    g = []
    for i in text:
        sp = i.split(":")
        g.append(len(sp[0]))
    return max(g)


def maxlenr(text):
    g = []
    for i in text:
        sp = i.split(":")
        g.append(len(sp[-1]))
    return max(g)


def display(text):
    for i in text:
        s = i.split(":")
        st = s[0] + " " * (maxlenl(text) - len(s[0])) + "    :    " \
             + s[-1] + " " * (maxlenr(text) - len(s[-1]))
        print
        st.center(167)


class Students(object):
    """A class for student's record"""

    def __init__(self):  # Initialising instance attributes
        self.Reg_No = 1
        self.Roll_No = None
        self.Name = None
        self.Father_Name = None
        self.Mother_Name = None
        self.Sex = None
        self.DOB = None
        self.Age = None
        self.Address = None
        self.Mobile_No = None
        self.Email = None
        self.Class = None
        self.Section = None
        self.Stream = None
        self.Fees = None
        self.MOC = None
        self.Fees_Months = 0
        self.Weight = None
        self.Height = None
        self.B_Group = None
        self.Aadhaar = None

    def get_fees_month(self):

        f_read = open("students.tc", "rb")
        try:
            while True:
                s = pickle.load(f_read)
                self.Fees_Months = s.Fees_Months + 1
        except EOFError:
            print

    def assign_reg_no(self):
        try:
            f_read = open("students.tc", "rb")
            try:
                while True:
                    s = pickle.load(f_read)
                    self.Reg_No = s.Reg_No + 1
            except EOFError:
                print
        except IOError:
            self.Reg_No = 1

    def assign_fees(self):  # Assigning Fees
        if self.Class <= 1:
            self.Fees = 3700
        elif 2 <= self.Class <= 6:
            self.Fees = 5500
        elif 7 <= self.Class <= 10:
            self.Fees = 6100
        elif 11 <= self.Class <= 12:
            self.Fees = 6700
        elif self.MOC == 'BUS':
            self.Fees += 1000
        elif self.MOC == 'AUTO':
            self.Fees += 400

    def get_data(self, *modify):  # Requesting data from user
        if modify == ():
            self.assign_reg_no()
            head("Registration Number - {0}".format(self.Reg_No))
        else:
            self.Reg_No = input("Enter New Registration Number ---> ")
        self.Roll_No = input("Enter Roll Number ---> ")
        self.Name = input("Enter Student's Name ---> ")
        self.Father_Name = input("Enter Father's Name ---> ")
        self.Mother_Name = input("Enter Mother's Name ---> ")
        self.Sex = input("Enter Sex [M/F] ---> ")
        self.DOB = input("Enter Date of Birth [DDMMYYYY] ---> ")
        self.Age = input("Enter Age ---> ")
        self.Address = input("Enter your address ---> ")
        self.Mobile_No = input("Enter Mobile Number ---> ")
        self.Email = input("Enter E-mail ---> ")
        self.Class = input("Enter Class ---> ")
        if self.Class in [11, 12]:
            self.Stream = input("Enter your stream [PMC, PBC, COMM, HUM] ---> ")
        else:
            self.Stream = 'N/A'
        self.Section = input("Enter Section ---> ")
        self.MOC = input("Enter Mode of Convenience [BUS, AUTO, SELF] ---> ")
        self.assign_fees()
        self.Weight = input("Enter your Weight [KG] ---> ")
        self.Height = input("Enter your Height [CM] ---> ")
        self.B_Group = input("Enter your Blood Group ---> ")
        self.Aadhaar = input("Enter Aadhaar number ---> ")

    def print_data(self):  # Displaying data to user

        head("Registration Number - {0}".format(self.Reg_No))
        l1 = "Roll Number : " + str(self.Roll_No)
        l2 = "Name : " + self.Name
        l3 = "Father's Name : " + self.Father_Name
        l4 = "Mother's Name : " + self.Mother_Name
        l5 = "Sex : " + self.Sex
        l6 = "Date of Birth : " + str(self.DOB)
        l7 = "Age : " + self.Age
        l8 = "Address : " + self.Address
        l9 = "Mobile Number : " + str(self.Mobile_No)
        l10 = "E-mail : " + self.Email
        l11 = "Class : " + str(self.Class)
        l12 = "Section : " + self.Section
        if self.Class in [11, 12]:
            l13 = "Stream : " + self.Stream
        else:
            l13 = "Stream : " + "N/A"
        l14 = "Mode of  Convenience : " + self.MOC
        l15 = "Fees applicable : " + str(self.Fees)
        l16 = "Fees paid of months : " + str(self.Fees_Months)
        l17 = "Weight : " + str(self.Weight)
        l18 = "Height : " + str(self.Height)
        l19 = "Blood Group : " + self.B_Group
        l20 = "Aadhaar number : " + str(self.Aadhaar)
        display([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11,
                 l12, l13, l14, l16, l17, l18, l19, l20])

    def pay_fees(self):

        fees_month = input("Enter no. of months to pay fees ---> ")
        output("Amount to be paid : " + str(self.Fees * fees_month))
        c_no = randint(1000, 9999)
        ch = input("Enter '{0}' to confirm and finally pay fees ---> ".format(c_no))
        if ch == c_no:
            self.Fees_Months += fees_month
            output("Fees deposited successfully of registration number : "
                   + str(self.Reg_No))
        else:
            output("Fees Not Deposited")


def post_data():
    st = Students()  # Creating object of the main class
    st.GET_DATA()
    f_append = open("students.tc", "ab")
    pickle.dump(st, f_append)
    output(st.Name + " Registered successfully")
    f_append.close()


def show_data():
    try:
        f_read = open("students.tc", "rb")
        while True:
            st = pickle.load(f_read)
            st.PRINT_DATA()
    except EOFError:
        output("No more records found")

    except IOError:
        output("School management system not started yet. Please register"
               " at least one student to get started")


def search():
    def search_reg_no(regno):

        f_read = open("students.tc", "rb")
        flag = False
        try:
            while True:
                st = pickle.load(f_read)
                if st.Reg_No == regno:
                    flag = True
                    st.PRINT_DATA()
        except EOFError:
            if not flag:
                output("Record not found")
        if flag:
            return st.Reg_No

    def search_name(names):

        f_read = open("students.tc", "rb")
        flag = False
        try:
            while True:
                st = pickle.load(f_read)
                if st.Name.lower() == names.lower():
                    flag = True
                    st.PRINT_DATA()
        except EOFError:
            if not flag:
                output("Record not found")

    def search_roll_no(rollno):

        f_read = open("students.tc", "rb")
        Flag = False
        try:
            while True:
                st = pickle.load(f_read)
                if st.Roll_No == rollno:
                    Flag = True
                    st.PRINT_DATA()
        except EOFError:
            if not Flag:
                output("Record not found")

    def search_mobile_no(mobileno):

        f_read = open("students.tc", "rb")
        Flag = False
        try:
            while True:
                st = pickle.load(f_read)
                if st.Mobile_No == mobileno:
                    Flag = True
                    st.PRINT_DATA()
        except EOFError:
            if not Flag:
                output("Record not found")

    def search_aadhaar(adhar):

        f_read = open("students.tc", "rb")
        Flag = False
        try:
            while True:
                st = pickle.load(f_read)
                if st.Aadhaar == adhar:
                    Flag = True
                    st.PRINT_DATA()
        except EOFError:
            if not Flag:
                output("Record not found")

    def email(emails):

        f_read = open("students.tc", "rb")
        Flag = False
        try:
            while True:
                st = pickle.load(f_read)
                if st.email.lower() == emails.lower():
                    Flag = True
                    st.PRINT_DATA()
        except EOFError:
            if not Flag:
                output("Record not found")

    while True:
        head("Search Menu")
        menu(["1. Search by Registration Number",
              "2. Search by Name",
              "3. Search by Roll Number",
              "4. Search by Mobile Number",
              "5. Search by Aadhaar Number",
              "6. Search by email",
              "7. Exit search menu"])
        usr_ch = input("Enter your choice [Search Menu] ---> ")
        if usr_ch == 1:
            reg_no = input("Enter registration number of student ---> ")
            search_reg_no(reg_no)
        elif usr_ch == 2:
            name = input("Enter name of student ---> ")
            search_name(name)
        elif usr_ch == 3:
            roll_no = input("Enter Roll Number of student ---> ")
            search_roll_no(roll_no)
        elif usr_ch == 4:
            mobile_no = input("Enter Mobile Number of student ---> ")
            search_mobile_no(mobile_no)
        elif usr_ch == 5:
            aadhaar = input("Enter Aadhaar Number of student ---> ")
            search_aadhaar(aadhaar)
        elif usr_ch == 6:
            email = input("Enter email of student ---> ")
            email(email)
        elif usr_ch == 7:
            break


def pay_fees():
    reg_no = input("Enter registration number of student to pay his fees ---> ")
    f_read = open("students.tc", "rb")
    f_post = open("students_temp.tc", "ab")
    Flag = False
    temp_st = Students()
    try:
        while True:
            st = pickle.load(f_read)
            if st.Reg_No == reg_no:
                output("Fees paying of: " + st.Name)
                Flag = True
                temp_st = st
                temp_st.PAY_FEES()
                pickle.dump(temp_st, f_post)
            else:
                pickle.dump(st, f_post)
    except EOFError:
        if not Flag:
            output("Record not found")
        f_read.close()
        f_post.close()
        os.remove("students.tc")
        os.rename("students_temp.tc", "students.tc")


def modify_data():
    reg_no = input("Enter registration number of student to modify its record ---> ")
    f_read = open("students.tc", "rb")
    f_post = open("students_temp.tc", "ab")
    Flag = False
    temp_st = Students()
    try:
        while True:
            st = pickle.load(f_read)
            if st.Reg_No == reg_no:
                Flag = True
                while True:
                    head("Data Modification Menu")
                    output("Modifying record of : " + st.Name)
                    menu(["1. Modify Registration Number",
                          "2. Modify Roll Number",
                          "3. Modify Name",
                          "4. Modify Father's Name",
                          "5. Modify Mother's Name",
                          "6. Modify Sex",
                          "7. Modify Date of Birth",
                          "8. Modify Age",
                          "9. Modify Address",
                          "10. Modify Mobile Number",
                          "11. Modify E-mail",
                          "12. Modify Class",
                          "13. Modify Section",
                          "14. Modify Stream",
                          "15. Modify Fees",
                          "16. Modify Mode of Convenience",
                          "17. Modify Number of Months of deposited fees",
                          "18. Modify Weight",
                          "19. Modify Height",
                          "20. Modify Blood Group",
                          "21. Modify Aadhaar",
                          "22. Modify all records",
                          "23. Exit Data Modification Menu and saving changes"])
                    usr_ch = input("Enter your choice [Modification Menu] ---> ")
                    if usr_ch == 1:
                        temp_st = st
                        temp_st.Reg_No = input("Enter new Registration Number ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 2:
                        temp_st = st
                        temp_st.Roll_No = input("Enter new Roll Number ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 3:
                        temp_st = st
                        temp_st.Name = input("Enter new Name ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 4:
                        temp_st = st
                        temp_st.Father_Name = input("Enter new Father's Name ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 5:
                        temp_st = st
                        temp_st.Mother_Name = input("Enter new Mother's Name ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 6:
                        temp_st = st
                        temp_st.Sex = input("Enter new SEX [M/F] ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 7:
                        temp_st = st
                        temp_st.DOB = input("Enter new Date of Birth [DDMMYYYY] ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 8:
                        temp_st = st
                        temp_st.Age = input("Enter new Age ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 9:
                        temp_st = st
                        temp_st.Address = input("Enter new Address ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 10:
                        temp_st = st
                        temp_st.Mobile_No = input("Enter new Mobile Number ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 11:
                        temp_st = st
                        temp_st.Email = input("Enter new E-Mail ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 12:
                        temp_st = st
                        temp_st.Class = input("Enter new Class ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 13:
                        temp_st = st
                        temp_st.Section = input("Enter new Section ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 14:
                        temp_st = st
                        temp_st.Stream = input("Enter new stream ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 15:
                        temp_st = st
                        temp_st.Fees = input("Enter new Fees ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 16:
                        temp_st = st
                        temp_st.MOC = input("Enter new Mode of Convenience ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 17:
                        temp_st = st
                        temp_st.Fees_Months = input("Enter update number "
                                                    "of months of fees paid ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 18:
                        temp_st = st
                        temp_st.Weight = input("Enter new Weight ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 19:
                        temp_st = st
                        temp_st.Height = input("Enter new Height ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 20:
                        temp_st = st
                        temp_st.B_Group = input("Enter new Blood Group ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 21:
                        temp_st = st
                        temp_st.Aadhaar = input("Enter new Aadhaar ---> ")
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 22:
                        temp_st.GET_DATA(1)
                        pickle.dump(temp_st, f_post)
                        output("Record Modified successfully")
                        break
                    elif usr_ch == 23:
                        break
                    else:
                        output("Please enter valid choice")

            else:
                pickle.dump(st, f_post)
    except EOFError:
        if not Flag:
            output("Record not found")
        f_read.close()
        f_post.close()
        os.remove("students.tc")
        os.rename("students_temp.tc", "students.tc")


def remove_data():
    reg_no = input("Enter registration number of student "
                   "to delete its record ---> ")
    f_read = open("students.tc", "rb")
    f_post = open("students_temp.tc", "ab")
    Flag = False
    try:
        while True:
            st = pickle.load(f_read)
            if st.Reg_No == reg_no:
                Flag = True
                output("Record deleted successfully")
            else:
                pickle.dump(st, f_post)
    except EOFError:
        if not Flag:
            output("Record not found")
        f_read.close()
        f_post.close()
        os.remove("students.tc")
        os.rename("students_temp.tc", "students.tc")


def main_menu():
    head("School Management System")
    while True:
        menu(
            ["1. Register a new student",
             "2. Read all student's record",
             "3. Search",
             "4. Pay Fees",
             "5. Modify Data",
             "6. Remove Record",
             "7. Exit"])
        usr_ch = input("Enter your choice [Main Menu] ---> ")
        if usr_ch == 1:
            post_data()
        elif usr_ch == 2:
            show_data()
        elif usr_ch == 3:
            search()
        elif usr_ch == 4:
            pay_fees()
        elif usr_ch == 5:
            modify_data()
        elif usr_ch == 6:
            remove_data()
        elif usr_ch == 7:
            os.exit(1)
        else:
            output("Enter valid Input")


if __name__ == "__main__":
    main_menu()
