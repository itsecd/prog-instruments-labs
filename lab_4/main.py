import sys
from loguru import logger
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow
from findValue import findValueDataset, findValueXY, findValueWeek, findValueYear
from logger_setup import setup_logger

setup_logger()


class MainWindow(QMainWindow):
    def __init__(self):
        """Initializes Main Window"""
        try:
            super(MainWindow, self).__init__()
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)
            self.init_UI()
            path = " "
            file_type = 0
        except Exception as exc:
            logger.error(f"UI setup error: {exc}")

    def init_UI(self):
        """Initializes UI"""
        self.setWindowTitle("Программа")
        self.ui.input_line.setPlaceholderText("2023-09-28")
        self.ui.output_line.setPlaceholderText("96.5")
        self.ui.pushButton.clicked.connect(self.returnValue)
        path = self.ui.pushButtonPath.clicked.connect(self.setPath)

    def setPath(self):
        """Sets a folder"""
        try:
            self.path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Folder"
            )
            logger.debug("Folder has been selected.")
        except Exception as exc:
            logger.error(f"Folder selection error: {exc}")

    def returnValue(self):
        """Returns value from files"""
        try:
            logger.debug("Searching for value...")
            input_value = self.ui.input_line.text()
            index = str(self.ui.comboBox.currentText())
            if index == "dataset":
                value = findValueDataset(self.path, input_value)
            elif index == "X и Y":
                value = findValueXY(self.path, input_value)
            elif index == "разбивка по неделям":
                value = findValueWeek(self.path, input_value)
            elif index == "разбивка по годам":
                value = findValueYear(self.path, input_value)
            else:
                logger.info(f"Value was not found in any dataset.")
            self.ui.output_line.setText(str(value))
        except Exception as exc:
            logger.error(f"Returning of value error: {exc}")


def application():
    """Runs application"""
    try:
        app = QtWidgets.QApplication([])
        application = MainWindow()
        application.show()
        logger.debug("Programm is running...")
        sys.exit(app.exec_())
        logger.debug("Programm is closed.")
    except Exception as exc:
        logger.error(f"Running application error: {exc}")


if __name__ == "__main__":
    application()
 