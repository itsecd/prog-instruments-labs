from curl_cffi import requests as cffi_requests

from config import BANK_CONFIGS, USER_AGENT
from logging_config import logger
from extractors import JsonLdExtractor
from processors import BankProcessor
from utils import save_results_to_file, save_by_product_type, print_final_statistics


def main():
    """Main execution function"""
    logger.info("Starting bank cards processing pipeline")
    all_cards_data = []
    json_ld_extractor = JsonLdExtractor()
    bank_processor = BankProcessor(json_ld_extractor)

    for bank_config in BANK_CONFIGS:
        bank_url = bank_config["url"]
        bank_name = bank_config["name"]
        product_type = bank_config["product_type"]

        logger.info("Processing bank: %s (%s)", bank_name, product_type)
        logger.debug("Bank URL: %s", bank_url)

        try:
            response = cffi_requests.get(bank_url, impersonate=USER_AGENT)
            response.raise_for_status()

            json_ld_data = json_ld_extractor.extract_from_html(response.text)
            card_urls = json_ld_extractor.extract_card_urls(
                json_ld_data, bank_name, product_type
            )
            logger.info("Found %d cards for %s (%s)", len(card_urls), bank_name, product_type)
            for url in sorted(card_urls):
                logger.debug("Card URL: %s", url)
            if card_urls:
                bank_results = bank_processor.process_all_cards(card_urls, product_type)
                all_cards_data.extend(bank_results)
                successful_in_bank = sum(1 for r in bank_results if r.get('success'))
                logger.info("Bank %s completed: %d/%d successful",
                            bank_name, successful_in_bank, len(bank_results))

        except Exception as e:
            logger.error("Failed to process bank %s: %s", bank_name, str(e))

    # Save results
    if all_cards_data:
        save_results_to_file(all_cards_data)
        save_by_product_type(all_cards_data)
        print_final_statistics(all_cards_data)

        successful_cards = sum(1 for card in all_cards_data if card.get('success'))
        logger.info("Pipeline completed. Total: %d cards, Successful: %d",
                    len(all_cards_data), successful_cards)
    else:
        logger.warning("No cards data collected - pipeline completed with zero results")


if __name__ == "__main__":
    main()
