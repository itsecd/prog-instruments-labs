import multiprocessing
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit,
                             QProgressBar, QSpinBox, QFormLayout, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal
import time
import matplotlib.pyplot as plt
from itertools import product
from typing import List, Optional, Generator
import hashlib
import json
import constants
from tqdm import tqdm
import unittest
import argparse


class CardNumberFinder:
    def __init__(self, hash_value: str = None, last_four: str = None, bins: List[str] = None, middle_len: int = None,
                 path: str = None):
        self.bins = bins or constants.ALFABANK_VISA_DEBIT_BINS
        self.last_four = last_four or constants.LAST_4_CHARACTERS_CARD
        self.middle_len = middle_len or constants.MIDDLE_LENGTH
        self.hash_value = hash_value or constants.HASH_VALUE
        self.path = path or constants.PATH_TO_SAVE

    def generate_and_check_cards(self, bin_prefix: str) -> List[str]:
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
        hashed = hashlib.sha3_224(card_number.encode()).hexdigest()
        return hashed == self.hash_value

    def find_matching_cards(self, num_processes: int) -> Optional[List[str]]:
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
        try:
            with open(self.path, "w") as f:
                json.dump({"matching_cards": card_numbers}, f, indent=2)
        except IOError as e:
            print(f"Error saving to file: {e}")


def benchmark(hash_value: str, last_four: str, bins: List[str], middle_len: int = 6):
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
    def setUp(self):
        self.finder = CardNumberFinder(
            hash_value="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0",
            last_four="1234",
            bins=["123456"],
            middle_len=2
        )

    def test_luhn_check(self):
        self.assertTrue(CardNumberFinder.luhn_check("4111111111111111"))
        self.assertFalse(CardNumberFinder.luhn_check("4111111111111112"))

    def test_check_hash(self):
        test_card = "123456001234"
        test_hash = hashlib.sha3_224(test_card.encode()).hexdigest()
        finder = CardNumberFinder(hash_value=test_hash, last_four="1234", bins=["123456"], middle_len=2)
        self.assertTrue(finder.check_hash(test_card))

    def test_generate_and_check_cards(self):
        test_card = "123456001234"
        test_hash = hashlib.sha3_224(test_card.encode()).hexdigest()
        finder = CardNumberFinder(hash_value=test_hash, last_four="1234", bins=["123456"], middle_len=2)
        results = finder.generate_and_check_cards("123456")
        self.assertIn(test_card, results)


class WorkerThread(QThread):
    finished = pyqtSignal(list)
    progress = pyqtSignal(int)
    message = pyqtSignal(str)

    def __init__(self, finder: CardNumberFinder, num_processes: int):
        super().__init__()
        self.finder = finder
        self.num_processes = num_processes

    def run(self):
        self.message.emit("Starting search...")
        matching_cards = self.finder.find_matching_cards(self.num_processes)
        self.finished.emit(matching_cards if matching_cards else [])


class CardFinderGUI(QMainWindow):
    def __init__(self):
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
        self.start_btn.setEnabled(True)
        if results:
            self.output_area.append("\nFound matching cards:")
            for card in results:
                self.output_area.append(f"{card[:6]}******{card[-4:]}")
        else:
            self.output_area.append("\nNo matching cards found.")
        self.progress_bar.setValue(100)


def parse_args():
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
