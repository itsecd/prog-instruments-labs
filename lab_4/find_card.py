import multiprocessing as mp
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm #taqaddum
from typing import Optional, Union
import hashlib

CORES = mp.cpu_count()
logger = logging.getLogger()
logger.setLevel('INFO')


def check_card(main_card: int, hash: str, bins: tuple, last_numbers: str) -> Union[str, bool]:
    """
    Check if a card number matches a given hash.
    Parameters:
        main_card (int): The main part of the card number.
        hash (str): The hash to compare the card number against.
        bins (tuple): Tuple of bin numbers.
        last_numbers (str): The last part of the card number.
    Returns:
        Union[str, bool]: The full card number if it matches the hash, False otherwise.
    """
    try:
        for card in bins:
            card = f'{card}{main_card:06d}{last_numbers}'
            if hashlib.blake2s(card.encode()).hexdigest() == hash:
                return card
    except Exception as e:
        logger.error(f'Error occurred while checking card: {e}')
    return False


def processing_card(hash: str, bins: list, last_numbers: str, pools=CORES) -> Optional[str]:
    """
    Process card numbers in parallel to find a matching hash.
    Parameters:
        hash (str): The hash to compare the card number against.
        bins (list): List of bin numbers.
        last_numbers (str): The last part of the card number.
        pools (int, optional): Number of processes to use. Defaults to CORES.
    Returns:
        Optional[str]: The full card number if found, None otherwise.
    """
    try:
        arguments = []
        for i in range(1000000):
            arguments.append((i, hash, bins, last_numbers))
        with mp.Pool(processes=pools) as p:
            for res in p.starmap(check_card, tqdm(arguments, desc='processes :', ncols=120)):
                if res:
                    p.terminate()
                    return
    except Exception as e:
        logger.error(f'Error occurred while processing card: {e}')
    return None


def luna_algorithm(card_number: str) -> bool:
    """
    Apply the Luhn algorithm to validate a card number.
    Parameters:
        card_number (str): The card number to validate.
    Returns:
        bool: True if the card number is valid, False otherwise.
    """
    try:
        tmp = list(map(int, card_number))[::-1]
        for i in range(1, len(tmp), 2):
            tmp[i] *= 2
            if tmp[i] > 9:
                tmp[i] = tmp[i] % 10 + tmp[i] // 10
        return sum(tmp) % 10 == 0
    except Exception as e:
        logger.error(f'Error occurred while applying Luhn algorithm: {e}')


def graphing_and_save(data: dict, filename: str) -> None:
    """
    Create a bar graph from data and save it to a file.
    Parameters:
        data (dict): Dictionary containing process pools and corresponding work times.
        filename (str): The name of the file to save the graph to.
    """
    try:
        fig = plt.figure(figsize=(30, 5))
        plt.ylabel('Time for working, s')
        plt.xlabel('Processes')
        plt.title('Graph')
        pools, work_times = data.keys(), data.values()
        plt.bar(pools, work_times, width=0.5)
        plt.plot(
            range(1, len(work_times)+1),
            work_times,
            linestyle=":",
            color="black",
            marker="x",
            markersize=10,
        )
        plt.savefig(filename)
        logging.info(f'Result save to the file {filename} success.')
    except Exception as e:
        logger.error(f'Error occurred while graphing and saving: {e}')