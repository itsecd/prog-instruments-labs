import sys

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import *

from script_1 import create_annotation
from script_2 import create_dataset2, create_annotation2
from script_3 import create_dataset3, create_annotation3
from script_5 import Iterator


class Window(QMainWindow):
    def __init__(self) -> None:
        """
        Конструктор

        Данная функция вызывает все необходимые методы для создания окна
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        super().__init__()

        self.initUI()
        self.initIterators()
        self.createActions()
        self.createMenuBar()
        self.createToolBar()

    def initUI(self) -> None:
        """
        Инициализация главного окна и кнопок

        Данная функция создает главный виджет и размещает кнопки по макету
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        self.center()
        self.setWindowTitle('Tiger&Leopard')
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        leopard_btn = QPushButton('Next Leopard', self)
        tiger_btn = QPushButton('Next Tiger', self)

        pixmap = QPixmap('img/main_photo.jpg')
        self.lbl = QLabel(self)
        self.lbl.setPixmap(pixmap)
        self.lbl.setAlignment(Qt.AlignCenter)

        hbox = QHBoxLayout()
        hbox.addSpacing(1)
        hbox.addWidget(tiger_btn)
        hbox.addWidget(leopard_btn)

        vbox = QVBoxLayout()
        vbox.addSpacing(1)
        vbox.addWidget(self.lbl)
        vbox.addLayout(hbox)

        self.centralWidget.setLayout(vbox)

        tiger_btn.clicked.connect(self.nextTiger)
        leopard_btn.clicked.connect(self.nextLeopard)

        self.folderpath = ' '

        self.showMaximized()

    def initIterators(self) -> None:
        """
        Создание итераторов

        Данная функция создает два объекта-итератора для показа изображений
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        self.tigers = Iterator('tiger', 'dataset')
        self.leopards = Iterator('leopard', 'dataset')

    def nextTiger(self) -> None:
        """
        Пока следующего экземпляра tiger

        Данная функция получает следующий экземпляр(путь к нему) изображения и размещает на главном окне
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        lbl_size = self.lbl.size()
        next_image = next(self.tigers)
        if next_image != None:
            img = QPixmap(next_image).scaled(
                lbl_size, aspectRatioMode=Qt.KeepAspectRatio)
            self.lbl.setPixmap(img)
            self.lbl.setAlignment(Qt.AlignCenter)
        else:
            self.initIterators()
            self.nextTiger()

    def nextLeopard(self) -> None:
        """
        Пока следующего экземпляра leopard

        Данная функция получает следующий экземпляр(путь к нему) изображения и размещает на главном окне
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        lbl_size = self.lbl.size()
        next_image = next(self.leopards)
        if next_image != None:
            img = QPixmap(next_image).scaled(
                lbl_size, aspectRatioMode=Qt.KeepAspectRatio)
            self.lbl.setPixmap(img)
            self.lbl.setAlignment(Qt.AlignCenter)
        else:
            self.initIterators()
            self.nextLeopard()

    def center(self) -> None:
        """
        Центрирование главного окна относительно экрана

        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        widget_rect = self.frameGeometry()
        pc_rect = QDesktopWidget().availableGeometry().center()
        widget_rect.moveCenter(pc_rect)
        self.move(widget_rect.center())

    def createMenuBar(self) -> None:
        """
        Создание строки меню

        Данная функция создает меню, подменю и добавляет к ним действия
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        menuBar = self.menuBar()

        self.fileMenu = menuBar.addMenu('&File')
        self.fileMenu.addAction(self.exitAction)
        self.fileMenu.addAction(self.changeAction)

        self.annotMenu = menuBar.addMenu('&Annotation')
        self.annotMenu.addAction(self.createAnnotAction)

        self.dataMenu = menuBar.addMenu('&Dataset')
        self.dataMenu.addAction(self.createData2Action)

    def createToolBar(self) -> None:
        """
        Создание тулбара

        Данная функция создает тулбар и связывает с ним действия
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        fileToolBar = self.addToolBar('File')
        fileToolBar.addAction(self.exitAction)

        annotToolBar = self.addToolBar('Annotation')
        annotToolBar.addAction(self.createAnnotAction)

    def createActions(self) -> None:
        """
        Создание действий, связанных с меню и тулбаром

        Данная функция создает действия и связывает их с методами класса или другими функциями
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        self.exitAction = QAction(QIcon('img/exit.png'), '&Exit')
        self.exitAction.triggered.connect(qApp.quit)

        self.changeAction = QAction(QIcon('img/change.png'), '&Change dataset')
        self.changeAction.triggered.connect(self.changeDataset)

        self.createAnnotAction = QAction(
            QIcon('img/csv.png'), '&Create annotation for current dataset')
        self.createAnnotAction.triggered.connect(self.createAnnotation)

        self.createData2Action = QAction(
            QIcon('img/new_dataset.png'), '&Create dataset2')
        self.createData2Action.triggered.connect(self.createDataset2)

        self.createData3Action = QAction(
            QIcon('img/new_dataset.png'), '&Create dataset3')
        self.createData3Action.triggered.connect(self.createDataset3)

    def createAnnotation(self) -> None:
        """
        Создание аннотации

        Данная функция создает аннотацию для текущего датасета
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        if 'dataset2' in str(self.folderpath):
            create_annotation2()
        elif 'dataset3' in str(self.folderpath):
            create_annotation3()
        elif 'dataset' in str(self.folderpath):
            create_annotation()

    def createDataset2(self) -> None:
        """
        Создание датасета №2

        Данная функция создает новый датасет, соединяя имя класса с порядковым номером
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        create_dataset2()
        self.dataMenu.addAction(self.createData3Action)

    def createDataset3(self) -> None:
        """
        Создание датасета №3

        Данная функция создает новый датасет с рандомными числами
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        create_dataset3()

    def changeDataset(self) -> None:
        """
        Изменение датасета

        Данная функция изменяет текущий датасет
        Parameters
        ----------
        self
        Returns
        -------
        None
        """
        reply = QMessageBox.question(self, 'Warning', f'Are you sure you want to change current dataset?\nCurrent dataset: {str(self.folderpath)}',
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.folderpath = self.folderpath = QFileDialog.getExistingDirectory(
                self, 'Select Folder')
        else:
            pass

    def closeEvent(self, event: QEvent) -> None:
        """
        Функция позволяет спросить пользователя, уверен ли он в том, что хочет закрыть окно

        Parameters
        ----------
        self
        event: 
            Событие, возникающе после нажатия на закрытие приложения
        Returns
        -------
        None
        """
        reply = QMessageBox.question(self, 'Warning', 'Are you sure to quit?',
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main() -> None:
    """
    Данная функция создает объект приложения
    Parameters
    ----------
    self
    Returns
    -------
    None
    """
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()