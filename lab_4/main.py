from read_and_write_file import read_json, write_card, load_statistics, write_statistics
from find_card import processing_card, luna_algorithm, graphing_and_save
import time
import multiprocessing as mp
import logging
import argparse

SETTINGS_FILE = 'settings.json'
logger = logging.getLogger()
logger.setLevel('INFO')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-set', '--settings', default = SETTINGS_FILE, type=str,
                        help='Allows you to use your own json file with paths"(Enter the path to the file)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-crd', '--card_find', type=int,
                       help='Searches for the card number using a hash, you need to specify the number of processes')
    group.add_argument('-sta', '--statistics',
                       help='It turns out statistics by selecting the card number on a different number of processes')
    group.add_argument('-lun', '--luhn_algorithm',
                       help='Checks the validity of the card number using the Luna algorithm')
    group.add_argument('-vis', '--visualize_statistics',
                       help='Creates a histogram based on available statistics')
    args = parser.parse_args()
    settings = read_json(args.settings)
    if settings:
        match args.card_find or args.statistics or args.luhn_algorithm or args.visualize_statistics:
            case args.card_find:
                logging.info("Searches for the card number using a hash")
                card_number = processing_card(settings['hash'], settings['bin'], settings['last_number'], args.card_find)
                if card_number:
                    logging.info(f"The card number was found successfully: {card_number}")
                    write_card(str(card_number), settings['card_number'])
                else:
                    logging.info("Couldn't find the card number")
            case args.statistics:
                logging.info("Statistics run.")
                for i in range(1, int(mp.cpu_count() * 1.5)):
                    t1 = time.time()
                    processing_card(settings['hash'], settings['bin'], settings['last_number'], i)
                    t2 = time.time()
                    write_statistics(i, t2 - t1, settings['csv_statistics'])
                logging.info("Statistics have been calculated successfully.")
            case args.luhn_algorithm:
                logging.info("Luna algorithm.")
                data = read_json(settings['card_number'])
                if luna_algorithm(str(data['card_number'])):
                    logging.info("The card number is valid")
                else:
                    logging.info("The card number is not valid")
            case args.visualize_statistics:
                graphing_and_save(load_statistics(settings['csv_statistics']), settings['png_statistics'])
                logging.info("The histogram has been created successfully.")
