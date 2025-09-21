import json
import multiprocessing
import time
from typing import Generator
import hashlib
import itertools

import matplotlib.pyplot as plt

import CONST


class CardFinder:
    def __init__(self):
        self.bins = CONST.SBERBANK_VISA_DEBIT_BINS
        self.hash = CONST.HASH
        self.last_four = CONST.LAST_FOUR
        self.json_res = CONST.JSON_RES


    def generate_possible_numbers(self, bin: str)-> Generator[str, None,None]:
        middle_length = 6
        for middle in itertools.product('0123456789', repeat=middle_length):
            yield bin + ''.join(middle) + self.last_four


    @staticmethod
    def check_card_hash(args: tuple[str,str])-> str | None:
        hash, card_number = args
        hashed = hashlib.blake2b(card_number.encode(), digest_size=64).hexdigest()
        return card_number if hash == hashed else None


    @staticmethod
    def get_num_processes()->int:
        return multiprocessing.cpu_count()


    def find_card_number(self, bin: str, num_processes=get_num_processes())-> str | None:
        result = None
        with multiprocessing.Pool(processes=num_processes) as pool:
            tasks = ((self.hash, num) for num in self.generate_possible_numbers(bin)) #создание генератора задач
            for res in pool.imap_unordered(self.check_card_hash, tasks):
                if res:
                    result = res
                    break
        return result

    def get_graph(self, time_data: list)->None:
        processes = list(range(1,int(self.get_num_processes()*1.5) + 1))
        plt.figure(figsize=(10, 5))
        plt.plot(processes, time_data)

        plt.title("Time dependence on the number of processes")
        plt.xlabel("Number of processes")
        plt.ylabel("Time (seconds)")

        plt.xticks(processes)
        plt.grid(True)

        min_time = min(time_data)
        min_processes = time_data.index(min_time) + 1
        plt.scatter(min_processes, min_time, color="red", label="Point of global minimum")
        plt.show()


    def get_report(self, result: str | None, bin: str)-> dict[str, str | None | int]:
        report = {
            "status": "success" if result else "fail",
            "card_number": result,
            "bin": bin,
            "last_four_numbers": self.last_four,
            "hash": self.hash,
            "processes_used": self.get_num_processes()
        }
        return report


    def write_report(self, report: dict[str, str | None | int])-> None:
        with open(self.json_res, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


    @staticmethod
    def luhn_check(card_number: str)->bool:

        numbers = [int(d) for d in reversed(card_number)]

        for i in range(1, len(card_number), 2):
            doubled = numbers[i] * 2
            numbers[i] = doubled - 9 if doubled > 9 else doubled

        return sum(numbers) % 10 == 0


    def get_json_data(self):
        with open(self.json_res, 'r') as file:
            data = json.load(file)
        return data


    def clear_report(self):
        with open(self.json_res, 'w') as file:
            json.dump({}, file)


def run_experiment(card_finder_path, num_processes):
    time_res = []

    with open(card_finder_path, 'r') as f:
        settings = json.load(f)

    finder = CardFinder()
    finder.hash = settings["hash"]
    finder.last_four = settings["last_four"]
    finder.json_res = settings["json_res"]

    for i in range(1, num_processes + 1):
        time_start = time.time()
        for bin in CONST.SBERBANK_VISA_DEBIT_BINS:
            if finder.find_card_number(bin, i):
                res = time.time() - time_start
                time_res.append(res)
                break

    finder.get_graph(time_res)
