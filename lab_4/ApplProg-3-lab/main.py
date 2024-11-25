import sys
import os
import shutil

import PyQt5
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QPushButton, QMessageBox, QDialog, QVBoxLayout, QLineEdit, QFileDialog, QGridLayout, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon, QFont

import sources.lab_2_materials.default_dataset_operations
import sources.lab_2_materials.united_dataset
import sources.lab_2_materials.mixed_dataset
import sources.lab_2_materials.iterator


class PopupWindow(QMessageBox):
    def __init__(self, title: str, text_inside: str, informative_text: str) -> None:
        """
        Constructor of PopupWindow class, which inherits methods of QMessageBox,
        displays a text explaining the problem that has appeared
        """
        error = QMessageBox()
        error.setWindowIcon(QIcon("sources/images/icon.png"))
        error.setFixedSize(200, 200)
        error.setWindowTitle(title)
        error.setIcon(QMessageBox.Warning)
        error.setText(text_inside)
        error.setInformativeText(informative_text)
        error.exec_()


class WindowWithRequest(QDialog):
    def __init__(self, title: str) -> None:
        """
        Constructor of WindowWithRequest class, which inherits methods of QDialog,
        prompts the user to enter a string for various purposes (which are explained in the header)

        Args:
            title (str): a title of creating window
        """
        self.request_window = QDialog()
        self.title = title
        self.request_window.setWindowTitle(self.title)
        self.request_window.setWindowIcon(QIcon("sources/images/icon.png"))
        self.request_window.setFixedSize(300, 100)
        self.file_name = QLineEdit(self.request_window)
        self.window_layout = QVBoxLayout(self.request_window)
        self.window_layout.addWidget(self.file_name)
        self.request_window.exec_()

    def checking_correctness(self) -> bool:
        """
        The method of WindowWithRequest class, which checks the correctness of the entered name
        """
        for i in self.file_name.text():
            if i in ["<", ">", "«", ":", "#", "%", "&", "{", "}", "\\", "*", "$", "!", "'", "/", "|"]:
                PopupWindow(
                    "Incorrect file name", "Invalid characters are present", "The set name is 'default'")
                self.set_text('default')
                return False
        return True

    def get_text(self) -> str:
        """
        The method of WindowWithRequest class, which return the text written in the window
        by the user
        """
        return self.file_name.text()

    def set_text(self, string: str) -> None:
        """
        Method of WindowWithRequest class, which allows you to specifically change the text 
        entered by the user in a non-standard situation

        Args:
            string (str): string with a new text
        """
        self.file_name.set_text(string)


class Interface(QMainWindow):
    def __init__(self) -> None:
        """
        Constructor of Interface class, which inherits methods of QMainWindow
        """
        super().__init__()
        self.initUI()

    def initUI(self) -> None:
        """
        The method of Interface class, which creates the user interface of the application.
        It defines the main entities that the user will interact with
        """
        self.setWindowTitle("ImageMaster")
        self.setWindowIcon(QIcon("sources/images/icon.png"))

        self.centralwidget = QWidget()
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        self.status = "free"

        files = os.listdir(os.getcwd())

        for file in files:
            if file.endswith(".csv"):
                os.remove(os.path.join(os.getcwd(), file))

        if os.path.isdir("new_dataset"):
            shutil.rmtree("new_dataset")

        self.window_image = QLabel(self)
        self.image = QPixmap("sources/images/test_image.png")
        self.image = self.image.scaled(1280, 720, QtCore.Qt.KeepAspectRatio)
        self.window_image.setPixmap(self.image)
        self.window_image.setAlignment(Qt.AlignCenter)

        self.choose_folder_button = QPushButton(
            text="Choose folder", clicked=self.choose_folder)
        self.choose_folder_button.setFont(QFont("IMPACT", 18))

        self.button_change_end = QPushButton(
            text="End", clicked=self.handle_end_button)
        self.button_change_end.setFont(QFont("IMPACT", 18))

        self.button_1 = QPushButton(
            text="Create ordered dataset", clicked=self.copy_dataset)
        self.button_1.setFont(QFont("IMPACT", 18))

        self.button_2 = QPushButton(
            text="Create mixed dataset (with annotation)", clicked=self.mixed_dataset)
        self.button_2.setFont(QFont("IMPACT", 16))

        self.button_create_annotation = QtWidgets.QPushButton(
            text="Create annotation", clicked=self.create_annotation)
        self.button_create_annotation.setFont(QFont("IMPACT", 15))

        self.previous1 = QPushButton(text="", clicked=self.option_previous1)
        self.previous1.setIcon(QtGui.QIcon("sources/images/left-arrow.png"))

        self.next1 = QPushButton(text="", clicked=self.option_next1)
        self.next1.setIcon(QtGui.QIcon("sources/images/right-arrow.png"))

        self.previous2 = QPushButton(text="", clicked=self.option_previous2)
        self.previous2.setIcon(QtGui.QIcon("sources/images/left-arrow.png"))

        self.next2 = QPushButton(text="", clicked=self.option_next2)
        self.next2.setIcon(QtGui.QIcon("sources/images/right-arrow.png"))

        self.previous1.setEnabled(False)
        self.previous2.setEnabled(False)
        self.next1.setEnabled(False)
        self.next2.setEnabled(False)
        self.button_create_annotation.setEnabled(False)
        self.button_change_end.setEnabled(False)

        self.file_path = QLabel(self)
        self.file_path.setText("PATH")
        self.file_path.setFont(QFont("IMPACT", 10))
        self.file_path.setStyleSheet("color: white;")
        self.file_path.setAlignment(Qt.AlignCenter)

        self.layout = QGridLayout(self.centralwidget)
        self.layout.addWidget(self.button_change_end, 0, 0, 1, 2)
        self.layout.addWidget(self.choose_folder_button, 0, 2, 1, 2)
        self.layout.addWidget(self.button_1, 0, 4, 1, 2)
        self.layout.addWidget(self.button_2, 0, 6, 1, 2)
        self.layout.addWidget(self.window_image, 1, 0, 4, 8)
        self.layout.addWidget(self.previous1, 5, 0)
        self.layout.addWidget(self.next1, 5, 1)
        self.layout.addWidget(self.previous2, 5, 2)
        self.layout.addWidget(self.next2, 5, 3)
        self.layout.addWidget(self.file_path, 5, 4, 1, 3)
        self.layout.addWidget(self.button_create_annotation, 5, 7)

        self.show()

    def create_annotation(self) -> None:
        """
        The method of Interface class, which is associated with the annotation creation 
        button for the dataset; creates annotations for the default dataset and the combined one, 
        the annotation for the random dataset is created by another button
        """
        try:
            window = WindowWithRequest("Creating the csv file")
            if window.checking_correctness() == True:
                sources.lab_2_materials.default_dataset_operations.create_file(
                    f"{window.get_text()}.csv")
                if self.status == "two folders":
                    sources.lab_2_materials.default_dataset_operations.input_data(
                        self.folderpath, f"{window.get_text()}.csv")
                if self.status == "ordered dataset":
                    sources.lab_2_materials.united_dataset.input_data(
                        self.folderpath, f"{window.get_text()}.csv")
        except AttributeError:
            self.button_create_annotation.click()
            

    def choose_folder(self) -> None:
        """
        The method of Interface class, which is associated with the button to select 
        the default dataset; when executed, it allows you to navigate through the elements 
        of the dataset in case of interaction with buttons like "next" and "previous"
        """
        try:
            self.status = "two folders"
            self.previous1.setEnabled(True)
            self.previous2.setEnabled(True)
            self.next1.setEnabled(True)
            self.next2.setEnabled(True)
            self.button_change_end.setEnabled(True)
            self.button_create_annotation.setEnabled(True)
            self.folderpath = QFileDialog.getExistingDirectory(
                self, "Select folder", "")
            self.iterator1 = sources.lab_2_materials.iterator.Iterator(
                os.path.join(self.folderpath, os.listdir(self.folderpath)[0]))
            self.iterator2 = sources.lab_2_materials.iterator.Iterator(
                os.path.join(self.folderpath, os.listdir(self.folderpath)[1]))
            image_path = next(self.iterator1)
            self.image = QPixmap(image_path)
            if not image_path.endswith(".jpg"):
                PopupWindow("Problem", "This action cannot be performed now",
                            "Please, choose another folder")
                self.previous1.setEnabled(False)
                self.previous2.setEnabled(False)
                self.next1.setEnabled(False)
                self.next2.setEnabled(False)
                self.button_change_end.setEnabled(False)
                self.button_create_annotation.setEnabled(False)
                self.image = QPixmap("sources/images/occasion.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)
            self.file_path.setText(image_path)
        except FileNotFoundError:
            PopupWindow("Problem", "This action cannot be performed now",
                        "Please, choose another folder")
            self.previous1.setEnabled(False)
            self.previous2.setEnabled(False)
            self.next1.setEnabled(False)
            self.next2.setEnabled(False)
            self.button_change_end.setEnabled(False)
            self.button_create_annotation.setEnabled(False)
            self.image = QPixmap("sources/images/occasion.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)
        except NotADirectoryError:
            PopupWindow("Problem", "This action cannot be performed now",
                        "Please, choose another folder")
            self.previous1.setEnabled(False)
            self.previous2.setEnabled(False)
            self.next1.setEnabled(False)
            self.next2.setEnabled(False)
            self.button_change_end.setEnabled(False)
            self.button_create_annotation.setEnabled(False)
            self.image = QPixmap("sources/images/occasion.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)

    def copy_dataset(self) -> None:
        """        
        The method of Interface class, which is associated with the button to select 
        the dataset, which will be converted into a dataset of combined elements 
        (which will differ only in name, they will not be distributed in folders);
        you cannot navigate through this dataset (there was no such requirement in the terms of reference)
        """
        try:
            self.status = "ordered dataset"
            self.previous1.setEnabled(False)
            self.previous2.setEnabled(False)
            self.next1.setEnabled(False)
            self.next2.setEnabled(False)
            self.button_change_end.setEnabled(False)
            self.button_create_annotation.setEnabled(True)
            self.image = QPixmap("sources/images/loading.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)
            self.file_path.setText("The dataset is creating")
            self.folderpath = QFileDialog.getExistingDirectory(
                self, "Select folder", "")
            window = WindowWithRequest("Create the united dataset")
            self.folderpath1 = QFileDialog.getExistingDirectory(
                self, "Select folder", "")
            self.folderpath = sources.lab_2_materials.united_dataset.copy_dataset(
                self.folderpath, f"{self.folderpath1}/{window.get_text()}")
            self.image = QPixmap("sources/images/OK.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)
            self.file_path.setText(os.path.abspath(
                f"{self.folderpath1}/{window.get_text()}"))
        except FileNotFoundError:
            PopupWindow("Problem", "This action cannot be performed now",
                        "Please, choose another folder")
            self.image = QPixmap("sources/images/occasion.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)
        except NotADirectoryError:
            PopupWindow("Problem", "This action cannot be performed now",
                        "Please, choose another folder")
            self.image = QPixmap("sources/images/occasion.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)

    def mixed_dataset(self) -> None:
        """
        The method of Interface class, which is associated with the button to select 
        the dataset, which will be converted into a dataset of combined elements 
        which will not differ even in name; you cannot navigate through this dataset 
        (there was no such requirement in the terms of reference)
        """
        try:
            self.status = "mixed dataset"
            self.previous1.setEnabled(False)
            self.previous2.setEnabled(False)
            self.next1.setEnabled(False)
            self.next2.setEnabled(False)
            self.button_change_end.setEnabled(False)
            self.button_create_annotation.setEnabled(False)
            self.image = QPixmap("sources/images/loading.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)
            self.file_path.setText("The dataset is creating")
            self.folderpath = QFileDialog.getExistingDirectory(
                self, "Select the parent folder", "")
            window = WindowWithRequest("Create the random dataset")
            self.folderpath1 = QFileDialog.getExistingDirectory(
                self, "Select the location of the dataset to be created", "")
            window1 = WindowWithRequest(
                "Create the annotation for random dataset")
            self.folderpath = sources.lab_2_materials.mixed_dataset.copy_and_rename_dataset(
                self.folderpath, f"{self.folderpath1}/{window.get_text()}", f"{window1.get_text()}")
            self.image = QPixmap("sources/images/OK.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)
            self.file_path.setText(os.path.abspath(
                f"{self.folderpath1}/{window.get_text()}"))
        except FileNotFoundError:
            PopupWindow("Problem", "This action cannot be performed now",
                        "Please, choose another a folder")
            self.image = QPixmap("sources/images/occasion.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)
        except NotADirectoryError:
            PopupWindow("Problem", "This action cannot be performed now",
                        "Please, choose another folder")
            self.previous1.setEnabled(False)
            self.previous2.setEnabled(False)
            self.next1.setEnabled(False)
            self.next2.setEnabled(False)
            self.button_change_end.setEnabled(False)
            self.button_create_annotation.setEnabled(False)
            self.image = QPixmap("sources/images/occasion.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)
        except PermissionError:
            PopupWindow("Problem", "You chose inappropriate folder",
                        "Please, choose another folder")
            self.image = QPixmap("sources/images/occasion.png")
            self.image = self.image.scaled(
                1280, 720, QtCore.Qt.KeepAspectRatio)
            self.window_image.setPixmap(self.image)
            self.window_image.setAlignment(Qt.AlignCenter)
            shutil.rmtree(f"{self.folderpath1}/{window.get_text()}")

    def option_next1(self) -> None:
        """
        The method Interface class, which is associated with the button to move 
        to the next element of the first folder of the transferred dataset,
        interacts with the imported Iterator class
        """
        if self.button_change_end.text() == "End":
            try:
                image_path = next(self.iterator1)
                self.image = QPixmap(image_path)
                if not image_path.endswith(".jpg"):
                    PopupWindow("Problem", "This action cannot be performed now",
                                "Please, choose another folder")
                    self.previous1.setEnabled(False)
                    self.previous2.setEnabled(False)
                    self.next1.setEnabled(False)
                    self.next2.setEnabled(False)
                    self.button_change_end.setEnabled(False)
                    self.button_create_annotation.setEnabled(False)
                    self.image = QPixmap("sources/images/occasion.png")
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)
            except StopIteration:
                self.iterator1.counter -= 1
                PopupWindow(
                    "Problem", "This action cannot be performed now", "There is no next element")
            except AttributeError:
                PopupWindow(
                    "Problem", "This action cannot be performed now", "There is no next element")
        if self.button_change_end.text() == "Cycle":
            try:
                image_path = next(self.iterator1)
                self.image = QPixmap(image_path)
                if not image_path.endswith(".jpg"):
                    PopupWindow("Problem", "This action cannot be performed now",
                                "Please, choose another folder")
                    self.previous1.setEnabled(False)
                    self.previous2.setEnabled(False)
                    self.next1.setEnabled(False)
                    self.next2.setEnabled(False)
                    self.button_change_end.setEnabled(False)
                    self.button_create_annotation.setEnabled(False)
                    self.image = QPixmap("sources/images/occasion.png")
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)
            except StopIteration:
                self.iterator1.counter -= self.iterator1.limit
                image_path = next(self.iterator1)
                self.image = QPixmap(image_path)
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)

    def option_next2(self) -> None:
        """
        The method Interface class, which is associated with the button to move 
        to the next element of the second folder of the transferred dataset,
        interacts with the imported Iterator class
        """
        if self.button_change_end.text() == "End":
            try:
                image_path = next(self.iterator2)
                self.image = QPixmap(image_path)
                if not image_path.endswith(".jpg"):
                    PopupWindow("Problem", "This action cannot be performed now",
                                "Please, choose another folder")
                    self.previous1.setEnabled(False)
                    self.previous2.setEnabled(False)
                    self.next1.setEnabled(False)
                    self.next2.setEnabled(False)
                    self.button_change_end.setEnabled(False)
                    self.button_create_annotation.setEnabled(False)
                    self.image = QPixmap("sources/images/occasion.png")
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)
            except StopIteration:
                self.iterator2.counter -= 1
                PopupWindow(
                    "Problem", "This action cannot be performed now", "There is no next element")
            except AttributeError:
                PopupWindow(
                    "Problem", "This action cannot be performed now", "There is no next element")
        if self.button_change_end.text() == "Cycle":
            try:
                image_path = next(self.iterator2)
                self.image = QPixmap(image_path)
                if not image_path.endswith(".jpg"):
                    PopupWindow("Problem", "This action cannot be performed now",
                                "Please, choose another folder")
                    self.previous1.setEnabled(False)
                    self.previous2.setEnabled(False)
                    self.next1.setEnabled(False)
                    self.next2.setEnabled(False)
                    self.button_change_end.setEnabled(False)
                    self.button_create_annotation.setEnabled(False)
                    self.image = QPixmap("sources/images/occasion.png")
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)
            except StopIteration:
                self.iterator2.counter -= self.iterator2.limit
                image_path = next(self.iterator2)
                self.image = QPixmap(image_path)
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)

    def option_previous1(self) -> None:
        """
        The method Interface class, which is associated with the button to move 
        to the previous element of the first folder of the transferred dataset,
        interacts with the imported Iterator class
        """
        if self.button_change_end.text() == "End":
            try:
                image_path = self.iterator1.previous()
                self.image = QPixmap(image_path)
                if not image_path.endswith(".jpg"):
                    PopupWindow("Problem", "This action cannot be performed now",
                                "Please, choose another folder")
                    self.previous1.setEnabled(False)
                    self.previous2.setEnabled(False)
                    self.next1.setEnabled(False)
                    self.next2.setEnabled(False)
                    self.button_change_end.setEnabled(False)
                    self.button_create_annotation.setEnabled(False)
                    self.image = QPixmap("sources/images/occasion.png")
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)
            except StopIteration:
                self.iterator1.counter += 1
                PopupWindow(
                    "Problem", "This action cannot be performed now", "There is no element behind")
            except AttributeError:
                PopupWindow(
                    "Problem", "This action cannot be performed now", "There is no element behind")
        if self.button_change_end.text() == "Cycle":
            try:
                image_path = self.iterator1.previous()
                self.image = QPixmap(image_path)
                if not image_path.endswith(".jpg"):
                    PopupWindow("Problem", "This action cannot be performed now",
                                "Please, choose another folder")
                    self.previous1.setEnabled(False)
                    self.previous2.setEnabled(False)
                    self.next1.setEnabled(False)
                    self.next2.setEnabled(False)
                    self.button_change_end.setEnabled(False)
                    self.button_create_annotation.setEnabled(False)
                    self.image = QPixmap("sources/images/occasion.png")
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)
            except StopIteration:
                self.iterator1.counter = self.iterator1.limit
                image_path = self.iterator1.previous()
                self.image = QPixmap(image_path)
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)

    def option_previous2(self) -> None:
        """
        The method Interface class, which is associated with the button to move 
        to the previous element of the second folder of the transferred dataset,
        interacts with the imported Iterator class
        """
        if self.button_change_end.text() == "End":
            try:
                image_path = self.iterator2.previous()
                self.image = QPixmap(image_path)
                if not image_path.endswith(".jpg"):
                    PopupWindow("Problem", "This action cannot be performed now",
                                "Please, choose another folder")
                    self.previous1.setEnabled(False)
                    self.previous2.setEnabled(False)
                    self.next1.setEnabled(False)
                    self.next2.setEnabled(False)
                    self.button_change_end.setEnabled(False)
                    self.button_create_annotation.setEnabled(False)
                    self.image = QPixmap("sources/images/occasion.png")
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)
            except StopIteration:
                self.iterator2.counter += 1
                PopupWindow(
                    "Problem", "This action cannot be performed now", "There is no element behind")
            except AttributeError:
                PopupWindow(
                    "Problem", "This action cannot be performed now", "There is no element behind")
        if self.button_change_end.text() == "Cycle":
            try:
                image_path = self.iterator2.previous()
                self.image = QPixmap(image_path)
                if not image_path.endswith(".jpg"):
                    PopupWindow("Problem", "This action cannot be performed now",
                                "Please, choose another folder")
                    self.previous1.setEnabled(False)
                    self.previous2.setEnabled(False)
                    self.next1.setEnabled(False)
                    self.next2.setEnabled(False)
                    self.button_change_end.setEnabled(False)
                    self.button_create_annotation.setEnabled(False)
                    self.image = QPixmap("sources/images/occasion.png")
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)
            except StopIteration:
                self.iterator2.counter = self.iterator2.limit
                image_path = self.iterator2.previous()
                self.image = QPixmap(image_path)
                self.image = self.image.scaled(
                    1280, 720, QtCore.Qt.KeepAspectRatio)
                self.window_image.setPixmap(self.image)
                self.window_image.setAlignment(Qt.AlignCenter)
                self.file_path.setText(image_path)

    def handle_end_button(self) -> None:
        """
        The method Interface class, which is associated with the button is an indicator 
        of the modifier for working with a default dataset
        """
        if self.button_change_end is not None:
            text = self.button_change_end.text()
            self.button_change_end.setText("Cycle" if text == "End" else "End")


stylesheet = """
    Interface {
        background-image: url(sources/images/background.jpg);
        background-repeat: no-repeat; 
        background-position: center;
    }
"""


def application() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(stylesheet)
    window = Interface()
    sys.exit(app.exec_())


if __name__ == "__main__":
    application()
