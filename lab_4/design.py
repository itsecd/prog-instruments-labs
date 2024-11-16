# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(367, 243)
        MainWindow.setStyleSheet("background-color: rgb(198, 255, 167);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.button_weather = QtWidgets.QPushButton(self.centralwidget)
        self.button_weather.setGeometry(QtCore.QRect(70, 60, 221, 41))
        self.button_weather.setStyleSheet("background-color: rgb(231, 255, 237);")
        self.button_weather.setObjectName("button_weather")
        self.text = QtWidgets.QLabel(self.centralwidget)
        self.text.setGeometry(QtCore.QRect(30, 10, 121, 41))
        self.text.setStyleSheet("font: 13pt \"MS Shell Dlg 2\";")
        self.text.setObjectName("text")
        self.button_chose_weather = QtWidgets.QDateEdit(self.centralwidget)
        self.button_chose_weather.setGeometry(QtCore.QRect(160, 10, 171, 41))
        self.button_chose_weather.setStyleSheet("background-color: rgb(231, 255, 237);")
        self.button_chose_weather.setObjectName("button_chose_weather")
        self.button_chose_weather.setCalendarPopup(True)
        self.button_chose_weather.setDate(QtCore.QDate.currentDate())
        self.button_chose_file = QtWidgets.QPushButton(self.centralwidget)
        self.button_chose_file.setGeometry(QtCore.QRect(20, 110, 111, 121))
        self.button_chose_file.setStyleSheet("background-color: rgb(231, 255, 237);")
        self.button_chose_file.setObjectName("button_chose_file")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(280, 110, 81, 131))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.get_folder_split_csv_x_y = QtWidgets.QPushButton(self.layoutWidget)
        self.get_folder_split_csv_x_y.setStyleSheet("background-color: rgb(231, 255, 237);")
        self.get_folder_split_csv_x_y.setObjectName("get_folder_split_csv_x_y")
        self.verticalLayout_2.addWidget(self.get_folder_split_csv_x_y)
        self.get_folder_button_split_csv_year = QtWidgets.QPushButton(self.layoutWidget)
        self.get_folder_button_split_csv_year.setStyleSheet("background-color: rgb(231, 255, 237);")
        self.get_folder_button_split_csv_year.setObjectName("get_folder_button_split_csv_year")
        self.verticalLayout_2.addWidget(self.get_folder_button_split_csv_year)
        self.get_folder_button_split_csv_weeks = QtWidgets.QPushButton(self.layoutWidget)
        self.get_folder_button_split_csv_weeks.setStyleSheet("background-color: rgb(231, 255, 237);")
        self.get_folder_button_split_csv_weeks.setObjectName("get_folder_button_split_csv_weeks")
        self.verticalLayout_2.addWidget(self.get_folder_button_split_csv_weeks)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(140, 110, 131, 131))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.split_csv_x_y = QtWidgets.QPushButton(self.widget)
        self.split_csv_x_y.setStyleSheet("background-color: rgb(231, 255, 237);")
        self.split_csv_x_y.setObjectName("split_csv_x_y")
        self.verticalLayout.addWidget(self.split_csv_x_y)
        self.split_csv_year = QtWidgets.QPushButton(self.widget)
        self.split_csv_year.setStyleSheet("background-color: rgb(231, 255, 237);")
        self.split_csv_year.setObjectName("split_csv_year")
        self.verticalLayout.addWidget(self.split_csv_year)
        self.split_csv_weeks = QtWidgets.QPushButton(self.widget)
        self.split_csv_weeks.setStyleSheet("background-color: rgb(231, 255, 237);")
        self.split_csv_weeks.setObjectName("split_csv_weeks")
        self.verticalLayout.addWidget(self.split_csv_weeks)
        MainWindow.setCentralWidget(self.centralwidget)


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Weather"))
        self.text.setText(_translate("MainWindow", "Выберите дату:"))
        self.button_weather.setText(_translate("MainWindow", "Узнать погоду"))
        self.button_chose_file.setText(_translate("MainWindow", "Выбрать файл"))
        self.get_folder_split_csv_x_y.setText(_translate("MainWindow", "Узнать погоду"))
        self.get_folder_button_split_csv_year.setText(_translate("MainWindow", "Узнать погоду"))
        self.get_folder_button_split_csv_weeks.setText(_translate("MainWindow", "Узнать погоду"))
        self.split_csv_x_y.setText(_translate("MainWindow", "Разделить на X и Y"))
        self.split_csv_year.setText(_translate("MainWindow", "Разделить по годам"))
        self.split_csv_weeks.setText(_translate("MainWindow", "Разделить по неделями"))
    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
