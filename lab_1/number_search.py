import argparse
import hashlib
import json
import multiprocessing
import time
import unittest
from itertools import product
from typing import List, Optional

import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm

import constants


class CardNumberFinder:
    """Class for finding card numbers that match a given hash value."""

    def __init__(self, hash_value: str = None, last_four: str = None, bins: List[str] = None, middle_len: int = None,
                 path: str = None):
        """
        Initialize the CardNumberFinder.
        :param hash_value:Target SHA3-224 hash value to match
        :param last_four:Last 4 digits of the card number
        :param bins:List of BIN (Bank Identification Number) prefixes
        :param middle_len:Length of the middle digits to brute force
        :param path:Length of the middle digits to brute force
        """
        self.bins = bins or constants.ALFABANK_VISA_DEBIT_BINS
        self.last_four = last_four or constants.LAST_4_CHARACTERS_CARD
        self.middle_len = middle_len or constants.MIDDLE_LENGTH
        self.hash_value = hash_value or constants.HASH_VALUE
        self.path = path or constants.PATH_TO_SAVE

    def generate_and_check_cards(self, bin_prefix: str) -> List[str]:
        """
        Generate card numbers for a given BIN and check against target hash.
        :param bin_prefix:BIN prefix to generate cards for
        :return:List of matching card numbers
        """
        matching = []
        total_possibilities = 10 ** self.middle_len
        with tqdm(product('0123456789', repeat=self.middle_len),
                  total=total_possibilities,
                  desc=f"Processing BIN {bin_prefix}",
                  leave=False) as pbar:
            for middle in pbar:
                card = bin_prefix + ''.join(middle) + self.last_four
                if self.check_hash(card):
                    matching.append(card)
                    pbar.set_postfix({'found': len(matching)})
        return matching

    @staticmethod
    def luhn_check(card_number: str) -> bool:
        """
        Validate card number using Luhn algorithm.
        :param card_number:Card number to validate
        :return:True if card number is valid, False otherwise
        """
        total = 0
        for i, digit in enumerate(reversed(card_number)):
            num = int(digit)
            if i % 2 == 1:
                num *= 2
                if num > 9:
                    num = (num // 10) + (num % 10)
            total += num
        return total % 10 == 0

    def check_hash(self, card_number: str) -> bool:
        """
        Check if card number matches the target hash.
        :param card_number:Card number to check
        :return:True if hash matches, False otherwise
        """
        hashed = hashlib.sha3_224(card_number.encode()).hexdigest()
        return hashed == self.hash_value

    def find_matching_cards(self, num_processes: int) -> Optional[List[str]]:
        """
        Find all card numbers that match the target hash using multiprocessing.
        :param num_processes:Number of parallel processes to use
        :return:List of matching card numbers or None if none found
        """
        matching_cards = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.imap_unordered(
                self.generate_and_check_cards,
                self.bins,
                chunksize=1
            )
            for result in tqdm(results, total=len(self.bins), desc="Processing BINs"):
                if result:
                    matching_cards.extend(result)
        return matching_cards or None

    def save_to_json(self, card_numbers: List[str]) -> None:
        """
        Save matching card numbers to JSON file.
        :param card_numbers:List of card numbers to save
        :return:None
        """
        try:
            with open(self.path, "w") as f:
                json.dump({"matching_cards": card_numbers}, f, indent=2)
        except IOError as e:
            print(f"Error saving to file: {e}")


def benchmark(hash_value: str, last_four: str, bins: List[str], middle_len: int = 6):
    """
    Benchmark performance with different numbers of processes.
    :param hash_value:Target hash value
    :param last_four:Last 4 digits of card
    :param bins:List of BIN prefixes
    :param middle_len:Length of middle digits
    :return:Tuple of (process_counts, times) for each process count
    """
    max_processes = int(multiprocessing.cpu_count() * 1.5)
    process_counts = range(1, max_processes + 1)
    times = []

    finder = CardNumberFinder(hash_value, last_four, bins, middle_len)

    print(f"Benchmarking with process counts from 1 to {max_processes}...")

    for num_processes in process_counts:
        start_time = time.time()
        matching_cards = finder.find_matching_cards(num_processes)
        elapsed = time.time() - start_time
        times.append(elapsed)

        if matching_cards:
            print(f"\nFound {len(matching_cards)} matching card(s) with {num_processes} processes:")
            for i, card in enumerate(matching_cards, 1):
                print(f"{i}. {card[:6]}******{card[-4:]}")
            finder.save_to_json(matching_cards)
            print(f"Results saved to {finder.path}")

    return process_counts, times


def plot_results(process_counts, times):
    """
    Plot benchmark results and find optimal number of processes.
    :param process_counts:List of process counts used
    :param times:List of execution times for each process count
    :return:Optimal number of processes
    """
    min_time = min(times)
    min_index = times.index(min_time)
    optimal_processes = process_counts[min_index]

    plt.figure(figsize=(10, 6))
    plt.plot(process_counts, times, 'b-o', label='Search time')
    plt.plot(optimal_processes, min_time, 'ro', label=f'Optimal ({optimal_processes} processes)')

    plt.title('Card Number Search Performance')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time (seconds)')
    plt.xticks(process_counts)
    plt.grid(True)
    plt.legend()

    plt.savefig('performance_plot.png')
    plt.show()

    return optimal_processes


class TestCardNumberFinder(unittest.TestCase):
    """Test cases for CardNumberFinder class."""

    def setUp(self):
        """
        Set up test fixtures.
        :return: None
        """
        self.finder = CardNumberFinder(
            hash_value="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",
            last_four="1234",
            bins=["123456"],
            middle_len=2
        )

    def test_luhn_check(self):
        """
        Test Luhn algorithm validation.
        :return:None
        """
        self.assertTrue(CardNumberFinder.luhn_check("4111111111111111"))
        self.assertFalse(CardNumberFinder.luhn_check("4111111111111112"))

    def test_check_hash(self):
        """
        Test hash checking functionality.
        :return: None
        """
        test_card = "123456001234"
        test_hash = hashlib.sha3_224(test_card.encode()).hexdigest()
        finder = CardNumberFinder(hash_value=test_hash, last_four="1234", bins=["123456"], middle_len=2)
        self.assertTrue(finder.check_hash(test_card))

    def test_generate_and_check_cards(self):
        """
        Test card generation and hash checking.
        :return: None
        """
        test_card = "123456001234"
        test_hash = hashlib.sha3_224(test_card.encode()).hexdigest()
        finder = CardNumberFinder(hash_value=test_hash, last_four="1234", bins=["123456"], middle_len=2)
        results = finder.generate_and_check_cards("123456")
        self.assertIn(test_card, results)


class WorkerThread(QThread):
    """Worker thread for running card search in background."""
    finished = pyqtSignal(list)
    progress = pyqtSignal(int)
    message = pyqtSignal(str)

    def __init__(self, finder: CardNumberFinder, num_processes: int):
        """
        Initialize worker thread.
        :param finder:CardNumberFinder instance
        :param num_processes:Number of processes to use
        """
        super().__init__()
        self.finder = finder
        self.num_processes = num_processes

    def run(self):
        """
        Execute the card search in the thread.
        :return: None
        """
        self.message.emit("Starting search...")
        matching_cards = self.finder.find_matching_cards(self.num_processes)
        self.finished.emit(matching_cards if matching_cards else [])


class CardFinderGUI(QMainWindow):
    """GUI application for card number finding."""

    def __init__(self):
        """Initialize the GUI application."""
        super().__init__()
        self.setWindowTitle("Card Number Finder")
        self.setGeometry(100, 100, 600, 500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        # Input fields
        self.form_layout = QFormLayout()

        self.hash_input = QLineEdit(constants.HASH_VALUE)
        self.last_four_input = QLineEdit(constants.LAST_4_CHARACTERS_CARD)
        self.bins_input = QLineEdit(",".join(constants.ALFABANK_VISA_DEBIT_BINS))
        self.middle_len_input = QSpinBox()
        self.middle_len_input.setValue(constants.MIDDLE_LENGTH)
        self.middle_len_input.setRange(1, 10)
        self.processes_input = QSpinBox()
        self.processes_input.setValue(multiprocessing.cpu_count())
        self.processes_input.setRange(1, multiprocessing.cpu_count() * 2)

        self.form_layout.addRow("Target Hash:", self.hash_input)
        self.form_layout.addRow("Last 4 Digits:", self.last_four_input)
        self.form_layout.addRow("BINs (comma separated):", self.bins_input)
        self.form_layout.addRow("Middle Length:", self.middle_len_input)
        self.form_layout.addRow("Processes:", self.processes_input)

        # Buttons
        self.start_btn = QPushButton("Start Search")
        self.start_btn.clicked.connect(self.start_search)

        self.benchmark_btn = QPushButton("Run Benchmark")
        self.benchmark_btn.clicked.connect(self.run_benchmark)

        # Output
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        self.layout.addLayout(self.form_layout)
        self.layout.addWidget(self.start_btn)
        self.layout.addWidget(self.benchmark_btn)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.output_area)

        self.central_widget.setLayout(self.layout)

        self.worker = None

    def start_search(self):
        """
        Start the card number search process.
        :return:None
        """
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Warning", "A search is already in progress!")
            return

        bins = [bin.strip() for bin in self.bins_input.text().split(",")]
        finder = CardNumberFinder(
            hash_value=self.hash_input.text(),
            last_four=self.last_four_input.text(),
            bins=bins,
            middle_len=self.middle_len_input.value(),
            path="found_cards.json"
        )

        self.worker = WorkerThread(finder, self.processes_input.value())
        self.worker.finished.connect(self.on_search_finished)
        self.worker.message.connect(self.output_area.append)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)

    def run_benchmark(self):
        """
        Run performance benchmark with different process counts.
        :return:None
        """
        bins = [bin.strip() for bin in self.bins_input.text().split(",")]
        self.output_area.append("Running benchmark...")

        QApplication.processEvents()  # Update UI

        process_counts, times = benchmark(
            self.hash_input.text(),
            self.last_four_input.text(),
            bins,
            self.middle_len_input.value()
        )

        optimal = plot_results(process_counts, times)
        self.output_area.append(f"Optimal number of processes: {optimal}")
        self.processes_input.setValue(optimal)

    def on_search_finished(self, results):
        """
        Handle search completion.
        :param results:Handle search completion.
        :return:None
        """
        self.start_btn.setEnabled(True)
        if results:
            self.output_area.append("\nFound matching cards:")
            for card in results:
                self.output_area.append(f"{card[:6]}******{card[-4:]}")
        else:
            self.output_area.append("\nNo matching cards found.")
        self.progress_bar.setValue(100)


def parse_args():
    """
    Parse command line arguments.
    :return:Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Card Number Finder")
    parser.add_argument("--hash", help="Target hash value", default=constants.HASH_VALUE)
    parser.add_argument("--last4", help="Last 4 digits of card", default=constants.LAST_4_CHARACTERS_CARD)
    parser.add_argument("--bins", help="Comma separated BINs",
                        default=",".join(constants.ALFABANK_VISA_DEBIT_BINS))
    parser.add_argument("--middle", type=int, help="Middle digits length",
                        default=constants.MIDDLE_LENGTH)
    parser.add_argument("--processes", type=int,
                        help="Number of processes", default=multiprocessing.cpu_count())
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    return parser.parse_args()


def main():
    """
    Main entry point of the application.
    :return: None
    """
    args = parse_args()

    if args.test:
        unittest.main(argv=[''], exit=False)
        return

    if args.gui:
        app = QApplication([])
        window = CardFinderGUI()
        window.show()
        app.exec_()
        return

    bins = [bin.strip() for bin in args.bins.split(",")]
    finder = CardNumberFinder(args.hash, args.last4, bins, args.middle)

    if args.benchmark:
        process_counts, times = benchmark(args.hash, args.last4, bins, args.middle)
        optimal = plot_results(process_counts, times)
        print(f"Optimal number of processes: {optimal}")
    else:
        matching_cards = finder.find_matching_cards(args.processes)
        if matching_cards:
            print("\nFound matching cards:")
            for i, card in enumerate(matching_cards, 1):
                print(f"{i}. {card[:6]}******{card[-4:]}")
            finder.save_to_json(matching_cards)
            print(f"Results saved to {finder.path}")
        else:
            print("No matching cards found.")


if __name__ == "__main__":
    main()
