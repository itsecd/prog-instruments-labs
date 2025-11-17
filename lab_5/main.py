from curl_cffi import requests as cffi_requests

from config import BANK_CONFIGS, MAIN_OUTPUT_FILE
from extractors import JsonLdExtractor
from processors import BankProcessor
from utils import save_results_to_file, save_by_product_type, print_final_statistics


def main():
    """Main execution function"""
    all_cards_data = []
    json_ld_extractor = JsonLdExtractor()
    bank_processor = BankProcessor(json_ld_extractor)

    for bank_config in BANK_CONFIGS:
        bank_url = bank_config["url"]
        bank_name = bank_config["name"]
        product_type = bank_config["product_type"]

        print(f"\n{'=' * 60}")
        print(f"ОБРАБАТЫВАЕМ БАНК: {bank_name}")
        print(f"ТИП ПРОДУКТА: {product_type}")
        print(f"URL: {bank_url}")
        print(f"{'=' * 60}")

        try:
            response = cffi_requests.get(bank_url, impersonate="safari15_5")
            response.raise_for_status()

            json_ld_data = json_ld_extractor.extract_from_html(response.text)
            card_urls = json_ld_extractor.extract_card_urls(
                json_ld_data, bank_name, product_type
            )

            print(f"Найдено {len(card_urls)} карт {bank_name} ({product_type}):")
            for url in sorted(card_urls):
                print(f"  - {url}")

            if card_urls:
                bank_results = bank_processor.process_all_cards(card_urls, product_type)
                all_cards_data.extend(bank_results)

        except Exception as e:
            print(f"Ошибка при обработке банка {bank_name} ({product_type}): {e}")

    # Save results
    if all_cards_data:
        save_results_to_file(all_cards_data)
        save_by_product_type(all_cards_data)
        print_final_statistics(all_cards_data)

        print(f"\nОБРАБОТКА ЗАВЕРШЕНА!")
        # ИСПРАВЛЕНИЕ: используем константу вместо хардкода
        print(f"Основной файл: {MAIN_OUTPUT_FILE} ({len(all_cards_data)} карт)")
    else:
        print("\nНе удалось собрать данные ни по одной карте")


if __name__ == "__main__":
    main()
