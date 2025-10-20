def add_student(mycursor, mydb):
    print("-------------------------")
    print("Add Student Information")
    print("-------------------------")
    var1 = input("Enter the sch no. of the student: ")
    var2 = input("Enter the name of the student: ")
    var3 = input("Enter the age of the student: ")
    var4 = input("Enter the email of the student: ")
    var5 = input("Enter the phone no. of the student: ")

    mycursor.execute(
        "insert into studentsinfo values('"
        + var1
        + "','"
        + var2
        + "','"
        + var3
        + "','"
        + var4
        + "','"
        + var5
        + "')"
    )
    mydb.commit()
    print("Data saved successfully!")
    input("Press Enter to continue!")


def view_students(mycursor):
    print("------------------------")
    print("--- Student Records ---")
    print("------------------------")
    mycursor.execute("select * from studentsinfo")
    for i in mycursor:
        print(i)
    input("Press Enter to continue!")


def search_student(mycursor):
    print("------------------------")
    print("--- Search Student ---")
    print("------------------------")
    a = input("Enter the sch no. of the student:")

    mycursor.execute("select * from studentsinfo where Sch_no=('" + a + "')")
    result = mycursor.fetchall()

    if len(result) == 0:
        print("Enter valid sch no.!")
    else:
        for i in result:
            print(i)
    input("Press Enter to continue!")


def update_student(mycursor, mydb):
    print("------------------------")
    print("--- Update Student ---")
    print("------------------------")
    var1 = input("Enter the sch no. of the student: ")
    var2 = input("Enter the name of the student: ")
    var3 = input("Enter the age of the student: ")
    var4 = input("Enter the email of the student: ")
    var5 = input("Enter the phone no. of the student: ")

    mycursor.execute(
        "update studentsinfo set name=('"
        + var1
        + "'),age=('"
        + var2
        + "'),email=('"
        + var3
        + "'),phone=('"
        + var4
        + "') where Sch_no=('"
        + var5
        + "')"
    )
    mydb.commit()
    print("Data updated successfully!")
    input("Press Enter to continue!")


def delete_student(mycursor, mydb):
    print("------------------------")
    print("--- Delete Student ---")
    print("------------------------")
    var1 = input("Enter the sch no. of the student:")
    mycursor.execute("delete from studentsinfo where sch_no=('" + var1 + "')")
    mydb.commit()
    print("Data deleted successfully!")
    input("Press Enter to continue!")