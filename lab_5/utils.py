import json
from typing import List, Dict

from config import MAIN_OUTPUT_FILE, DEBIT_CARDS_FILE, CREDIT_CARDS_FILE
from logging_config import logger


def save_results_to_file(results: List[Dict], filename: str = MAIN_OUTPUT_FILE):
    """Save results to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Data saved to file: %s (%d records)", filename, len(results))
    except Exception as e:
        logger.error("Failed to save data to %s: %s", filename, e)


def save_by_product_type(all_cards_data: List[Dict]):
    """Save data separated by product type"""
    debit_cards = [card for card in all_cards_data if card.get('product_type') == 'debitcards']
    credit_cards = [card for card in all_cards_data if card.get('product_type') == 'creditcards']

    if debit_cards:
        save_results_to_file(debit_cards, DEBIT_CARDS_FILE)
        print(f"Дебетовые карты сохранены в: {DEBIT_CARDS_FILE}")
    if credit_cards:
        save_results_to_file(credit_cards, CREDIT_CARDS_FILE)
        print(f"Кредитные карты сохранены в: {CREDIT_CARDS_FILE}")


def print_final_statistics(all_cards_data: List[Dict]):
    """Print final statistics"""
    total_cards = len(all_cards_data)
    successful_cards = sum(1 for card in all_cards_data if card.get('success'))
    failed_cards = total_cards - successful_cards
    logger.info("FINAL STATISTICS: Total=%d, Successful=%d, Failed=%d",
                total_cards, successful_cards, failed_cards)
    debit_cards = [card for card in all_cards_data if card.get('product_type') == 'debitcards']
    credit_cards = [card for card in all_cards_data if card.get('product_type') == 'creditcards']

    banks = {}
    for card in all_cards_data:
        if card.get('success'):
            bank_name = card.get('bank')
            if bank_name:
                if bank_name not in banks:
                    banks[bank_name] = {'debit': 0, 'credit': 0}
                if card.get('product_type') == 'debitcards':
                    banks[bank_name]['debit'] += 1
                else:
                    banks[bank_name]['credit'] += 1

    print(f"ИТОГОВАЯ СТАТИСТИКА")
    print(f"Всего карт собрано: {total_cards}")
    print(f"Успешно обработано: {successful_cards}")
    print(f"С ошибками: {failed_cards}")
    print(f"Дебетовых карт: {len(debit_cards)}")
    print(f"Кредитных карт: {len(credit_cards)}")

    print(f"\nСТАТИСТИКА ПО БАНКАМ:")
    for bank_name, stats in sorted(banks.items()):
        total = stats['debit'] + stats['credit']
        print(f"  {bank_name}: {total} карт (дебетовых: {stats['debit']}, кредитных: {stats['credit']})")
