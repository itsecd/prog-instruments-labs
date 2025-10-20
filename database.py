import mysql.connector
from settings import settings


def init_database():
    mydb = mysql.connector.connect(
        host=settings['host'],
        user=settings['user'],
        passwd=settings['password']
    )
    mycursor = mydb.cursor()

    mycursor.execute("create database if not exists students")
    mycursor.execute("use students")
    mycursor.execute(
        "create table if not exists studentsinfo(sch_no int(50),name char(100),age int(50),email varchar(150),phone varchar(150))"
    )
    mycursor.execute(
        "create table if not exists admin(id char(100),password varchar(150))"
    )
    mydb.commit()

    return mydb, mycursor