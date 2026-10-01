"""Module containing the Graphical User Interface for the application."""
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QComboBox, QTextEdit, QSpinBox, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt
from src.hash_logic.statistics import run_experiments
from src.utils.file_manager import save_text_file

MODERN_STYLE = """
QMainWindow {
    background-color: #2b2b2b;
}
QLabel {
    color: #e0e0e0;
    font-size: 14px;
    font-weight: 500;
}
QLabel#TitleLabel {
    font-size: 22px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 10px;
}
QComboBox, QSpinBox {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 5px;
    padding: 8px;
    font-size: 14px;
}
QComboBox QAbstractItemView {
    background-color: #3c3c3c;
    color: #ffffff;
    selection-background-color: #0d6efd;
    selection-color: #ffffff;
    border: 1px solid #555555;
    outline: none; 
}
QPushButton {
    background-color: #0d6efd;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 12px 20px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #0b5ed7;
}
QPushButton:pressed {
    background-color: #0a58ca;
}
QPushButton:disabled {
    background-color: #555555;
    color: #888888;
}
QPushButton#SaveBtn {
    background-color: #198754;
}
QPushButton#SaveBtn:hover {
    background-color: #157347;
}
QPushButton#SaveBtn:pressed {
    background-color: #146c43;
}
QTextEdit {
    background-color: #1e1e1e;
    color: #00ff00;
    border: 1px solid #444444;
    border-radius: 6px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 14px;
    padding: 10px;
}
"""

class MainWindow(QMainWindow):
    """Main window class for the hash collision application."""

    def __init__(self, settings: dict):
        """
         Initializes the GUI components using provided settings.

         Args:
             settings (dict): Application settings dictionary loaded from JSON.

         Returns:
             None

         Raises:
             Exception: If an error occurs during GUI initialization.
         """
        try:
            super().__init__()
            self.settings = settings

            gui_cfg = settings["gui"]
            self.setWindowTitle(gui_cfg["window_title"])
            self.resize(gui_cfg["window_width"], gui_cfg["window_height"])

            self.setStyleSheet(MODERN_STYLE)

            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(20, 20, 20, 20)
            main_layout.setSpacing(15)

            title = QLabel(gui_cfg["window_title"])
            title.setObjectName("TitleLabel")
            title.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(title)

            settings_layout = QHBoxLayout()

            bits_layout = QVBoxLayout()
            bits_layout.addWidget(QLabel("Select Hash Bits:"))
            self.combo_bits = QComboBox()
            allowed_bits = [str(b) for b in settings["hasher"]["allowed_bits"]]
            self.combo_bits.addItems(allowed_bits)
            bits_layout.addWidget(self.combo_bits)
            settings_layout.addLayout(bits_layout)

            exp_layout = QVBoxLayout()
            exp_cfg = settings["experiments"]
            exp_layout.addWidget(QLabel("Number of Experiments:"))
            self.spin_exp = QSpinBox()
            self.spin_exp.setMinimum(exp_cfg["min_count"])
            self.spin_exp.setMaximum(exp_cfg["max_count"])
            self.spin_exp.setValue(exp_cfg["default_count"])
            exp_layout.addWidget(self.spin_exp)
            settings_layout.addLayout(exp_layout)

            main_layout.addLayout(settings_layout)

            btn_layout = QHBoxLayout()
            btn_layout.setSpacing(15)

            self.btn_start = QPushButton("Find Collisions")
            self.btn_start.clicked.connect(self.run_search)
            btn_layout.addWidget(self.btn_start)

            self.btn_save = QPushButton("Save Log to File")
            self.btn_save.setObjectName("SaveBtn")
            self.btn_save.clicked.connect(self.save_log)
            self.btn_save.setEnabled(False)
            btn_layout.addWidget(self.btn_save)

            main_layout.addLayout(btn_layout)

            self.text_log = QTextEdit()
            self.text_log.setReadOnly(True)
            self.text_log.setPlaceholderText("Program output will appear here...")
            main_layout.addWidget(self.text_log)

            central_widget = QWidget()
            central_widget.setLayout(main_layout)
            self.setCentralWidget(central_widget)
        except Exception as e:
            print(f"Error initializing GUI: {e}")

    def run_search(self):
        """
        Executes the hash collision search when the start button is clicked.

        Retrieves user input, runs the experiments, and updates the text log
        with the theoretical and practical results.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: If an unexpected error occurs during the search execution.
        """
        try:
            self.btn_start.setEnabled(False)
            self.btn_save.setEnabled(False)
            bits = int(self.combo_bits.currentText())
            count = self.spin_exp.value()

            self.text_log.append(f"\n[+] Starting {count} experiment(s) for {bits}-bit hash...")
            self.text_log.repaint()

            result = run_experiments(bits, count, self.settings)

            self.text_log.append(f"[*] Theoretical Expected Attempts: {result['theoretical_attempts']:.2f}")
            self.text_log.append(f"[*] Practical Average Attempts: {result['average_attempts']:.2f}\n")

            for idx, col in enumerate(result['collisions'], 1):
                self.text_log.append(
                    f"  Exp {idx}: '{col['str1']}' and '{col['str2']}'\n"
                    f"  -> Hash: {col['hash']} (Attempts: {col['attempts']})\n"
                )
        except Exception as e:
            self.text_log.append(f"[!] Error occurred: {str(e)}")
        finally:
            self.btn_start.setEnabled(True)
            self.btn_save.setEnabled(True)

    def save_log(self):
        """
        Saves the current content of the text log to a local text file.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: If an error occurs while writing to the file.
        """
        try:
            content = self.text_log.toPlainText()
            filename = "collision_log.txt"
            save_text_file(filename, content)

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"Log successfully saved to '{filename}'.")
            msg.setWindowTitle("Success")
            msg.setStyleSheet("QMessageBox { background-color: #2b2b2b; color: white; } "
                              "QLabel { color: white; } "
                              "QPushButton { background-color: #0d6efd; color: white; "
                              "padding: 5px 15px; border-radius: 4px; }")
            msg.exec_()
        except Exception as e:
            self.text_log.append(f"[!] Failed to save log: {e}")

def start_gui(settings: dict):
    """
    Initializes the QApplication and starts the PyQt5 main event loop.

    Args:
        settings (dict): Application settings dictionary to configure the window.

    Returns:
        None

    Raises:
        Exception: If the application fails to initialize or start.
    """
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        window = MainWindow(settings)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        raise Exception(f"Failed to start GUI: {e}")