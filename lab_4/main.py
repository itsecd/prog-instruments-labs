from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QComboBox)
import sys
from findValue import findValueDataset, findValueXY, findValueWeek, findValueYear
import matplotlib.pyplot as plt


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_UI()
        path = " "
        file_type = 0
        
    def init_UI(self):
        self.setWindowTitle('Программа')

        self.ui.input_line.setPlaceholderText('2023-09-28')
        self.ui.output_line.setPlaceholderText('96.5')
        self.ui.pushButton.clicked.connect(self.returnValue)
        path = self.ui.pushButtonPath.clicked.connect(self.setPath)

    """метод для указывания пути к датасетам"""
    def setPath(self):
        self.path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')

    """метод поиска значения"""
    def returnValue(self):
        input_value = self.ui.input_line.text()
        index = str(self.ui.comboBox.currentText())
        if (index == "dataset"):
            value = findValueDataset(self.path, input_value)
        elif (index == "X и Y"):
            value = findValueXY(self.path, input_value)
        elif (index == "разбивка по неделям"):
            value = findValueWeek(self.path, input_value)
        elif (index == "разбивка по годам"):
            value = findValueYear(self.path, input_value)
        self.ui.output_line.setText(str(value))

def application():
    app = QtWidgets.QApplication([])

    application = MainWindow()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    application()