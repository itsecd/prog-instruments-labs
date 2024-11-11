import logging
import json
import csv #Comma Separated Values

logger = logging.getLogger()
logger.setLevel('INFO')


def read_json(setting_file: str) -> dict:
    """
   Load settings from a JSON file.
   Parameters:
       setting_file (str): Path to the JSON file containing settings.
   Returns:
       dict: A dictionary containing settings loaded from the JSON file.
   Raises:
       OSError: If there is an error reading the settings file.
   """
    settings = None
    try:
        with open(setting_file) as f:
            settings = json.load(f)
        logging.info('Successfully reading the settings')
    except Exception as e:
        logging.error(f'Error reading setttings file: {e}')
    return settings


def write_card(card: str, filename: str) -> None:
    """
     Write a card number to a JSON file.
     Parameters:
         card (str): The card number to write.
         filename (str): The name of the file to write the card number to.
     """
    try:
        with open(filename, "w") as f:
            json.dump({"card_number": card}, f)
    except Exception as e:
        logging.error(f'Error writing text a file {filename}: {e}')


def load_statistics(filename: str) -> dict:
    """
    Load statistics from a CSV file.
    Parameters:
        filename (str): The name of the CSV file containing statistics.
    Returns:
        dict: A dictionary containing statistics loaded from the CSV file.
    """
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            stats = list(reader)
        logging.info("Statistics successfully read.")
    except Exception as e:
        logging.warning(f"Statistics could not be read {filename}: {e}")
    result = dict()
    for i in stats:
        processes, time = i
        result[int(processes)] = float(time)
    return result


def write_statistics(processes: int, time: float, filename: str) -> None:
    """
    Write statistics to a CSV file.
    Parameters:
        processes (int): The number of processes.
        time (float): The time taken for the processes.
        filename (str): The name of the CSV file to write the statistics to.
    """
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([processes, time])
        logging.info(f"Statistics written to a file {filename}")
    except Exception as e:
        logging.warning(f"Error writting statistics a file {filename}: {e} ")