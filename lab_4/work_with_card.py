import multiprocessing
import hashlib
import time
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from work_with_file import *
from logging_config import setup_logging

setup_logging()


def get_card_num(hash_str: str, last_nums: str, bin_str: str) -> List[str]:
    """
        Generate valid card numbers based on the given hash, last 4 numbers, and BIN.

        Args:
            hash_str (str): The hash to match.
            last_nums (str): The last 4 numbers of the card.
            bin_str (str): The BIN (Bank Identification Number) of the card.

        Returns:
            List[str]: List of valid card numbers.
    """

    logging.info(f"Generating valid card numbers for hash: {hash_str}, last 4 digits: {last_nums}, BIN: {bin_str}")
    valid_card_numbers = []
    for middle_num in range(0, 999999 + 1):
        card_num = f"{bin_str}{middle_num:06d}{last_nums}"
        if hashlib.sha3_256(card_num.encode()).hexdigest() == hash_str:
            valid_card_numbers.append(card_num)
            logging.debug(f"Found valid card number: {card_num}")
    logging.info(f"Found {len(valid_card_numbers)} valid card numbers")
    return valid_card_numbers


def find_num(hash_str: str, last_nums: str, bins: List[int], path_to: str) -> List[str]:
    """
    Find and save valid card numbers based on the given parameters.

    Args:
        hash_str (str): The hash to match.
        last_nums (str): The last 4 numbers of the card.
        bins (List[int]): List of BINs to search for valid card numbers.
        path_to (str): The path to save the results.

    Returns:
        List[str]: List of valid card numbers found.
    """

    logging.info(f"Finding valid card numbers for hash: {hash_str}, last 4 digits: {last_nums}, BINs: {bins}")
    ans = []
    args = [(hash_str, last_nums, str(bin_num)) for bin_num in bins]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for res in pool.starmap(get_card_num, args):
            ans += res

    logging.info(f"Found {len(ans)} valid card numbers")
    write_file(path_to, "\n".join(ans))
    return ans


def algorithm_luhn(card_num: str) -> bool:
    """
    Validate a credit card number using the Luhn algorithm.

    Args:
        card_num (str): The credit card number to validate.

    Returns:
        bool: True if the credit card number is valid, False otherwise.
    """
    logging.debug(f"Validating card number: {card_num}")
    total_sum = 0
    second_elem = False

    for elem in reversed(card_num):
        elem = int(elem)

        if second_elem:
            elem *= 2
            if elem > 9:
                elem -= 9

        total_sum += elem
        second_elem = not second_elem

    is_valid = total_sum % 10 == 0
    logging.info(f"Card number {card_num} is {'valid' if is_valid else 'invalid'}")
    return is_valid


def execute_processes(args, num_processes, start):
    """
    Execute multiple processes in parallel and calculate the time taken by each process.

    Args:
        args: Arguments to be passed to the processes.
        num_processes (int): Number of processes to execute.
        start: Start time of the processes.

    Returns:
        List[float]: List of times taken by each process.
    """

    logging.info(f"Executing {num_processes} processes in parallel")
    process_times = []
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.starmap(get_card_num, args)
        for result in results:
            process_time = time.time() - start
            process_times.append(process_time)
            logging.debug(f"Process completed in {process_time:.2f} seconds")
    return process_times


def get_stats(hash: str, last_nums: str, bins: List[int], path_to: str) -> List[float]:
    """
    Get statistics on the collision search time for different processes.

    Args:
        hash (str): The hash value.
        last_nums (str): The last numbers.
        bins (List[int]): List of bins.
        path_to (str): Path to save the statistics.

    Returns:
        List[float]: List of average times for each process.
    """

    logging.info(
        f"Gathering statistics on collision search time for hash: {hash}, last 4 digits: {last_nums}, BINs: {bins}")
    times = []

    for i in tqdm(range(1, int(1.5 * multiprocessing.cpu_count()) + 1), desc='State'):
        start = time.time()
        args = [(hash, last_nums, str(bin)) for bin in bins]
        process_times = execute_processes(args, i, start)
        avg_time = sum(process_times) / len(process_times)
        times.append(avg_time)
        logging.info(f"Average time for {i} processes: {avg_time:.2f} seconds")

    write_file(path_to, "\n".join(map(str, times)))
    return times


def graph(data: List[float], path_to_save: str):
    """
    Generate and save a graph based on the given data.

    Args:
        data (List[float]): List of data points.
        path_to_save (str): Path to save the graph image.

    Returns:
        None
    """
    logging.info(f"Generating graph and saving to {path_to_save}")
    plt.plot(range(1, len(data) + 1), data)

    m = min(data)
    m_idx = data.index(m)
    plt.scatter([m_idx + 1], [m], color='red')
    plt.annotate(f"Global minimum point ({m_idx + 1}, {round(m, 2)})", (m_idx + 1, m))

    plt.xlabel('Num of processes')
    plt.ylabel('Collision search time, s')
    plt.title('Search Time vs. Num of Processes')

    plt.savefig(path_to_save)
    logging.info(f"Graph saved to {path_to_save}")
    plt.show()
