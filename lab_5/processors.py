import time
from typing import List, Dict, Any, Set

from curl_cffi import requests as cffi_requests

from config import REQUEST_DELAY, BATCH_DELAY, USER_AGENT
from extractors import DataModuleExtractor, DataCleaner
from logging_config import logger


class CardProcessor:
    """Processes individual cards"""

    def __init__(self, delay: int = REQUEST_DELAY):
        self.delay = delay
        self.data_extractor = DataModuleExtractor()
        self.data_cleaner = DataCleaner()

    def process_single_card(self, card_url: str, product_type: str) -> Dict[str, Any]:
        """Process a single card"""
        try:
            logger.debug("Processing card: %s", card_url)
            time.sleep(self.delay)

            response = cffi_requests.get(card_url, impersonate=USER_AGENT)
            response.raise_for_status()
            logger.debug("Successfully fetched card page: %s", card_url)

            raw_data = self.data_extractor.extract(response.text)
            if not raw_data:
                logger.warning("No data extracted from card: %s", card_url)
                return self._create_error_result(card_url, 'no data')

            clean_data = self.data_cleaner.clean(raw_data, product_type)
            if not clean_data:
                logger.warning("Failed to clean data for card: %s", card_url)
                return self._create_error_result(card_url, 'failed to clean data')

            result = {
                'url': card_url,
                **clean_data,
                'success': True
            }

            logger.info("Successfully processed card: %s", clean_data['name'])
            return result

        except Exception as e:
            logger.error("Failed to process card %s: %s", card_url, str(e))
            return self._create_error_result(card_url, str(e))

    def _create_error_result(self, card_url: str, error: str) -> Dict[str, Any]:
        """Create standardized error result"""
        logger.debug("Creating error result for card: %s", card_url)
        return {
            'url': card_url,
            'success': False,
            'error': error
        }


class BankProcessor:
    """Processes bank data"""

    def __init__(self, json_ld_extractor, batch_delay: int = BATCH_DELAY):
        self.json_ld_extractor = json_ld_extractor
        self.card_processor = CardProcessor()
        self.batch_delay = batch_delay

    def process_all_cards(self, card_urls: Set[str], product_type: str) -> List[Dict]:
        """Process multiple card URLs"""
        logger.info("Starting batch processing of %d %s cards", len(card_urls), product_type)

        all_results = []
        total_count = len(card_urls)

        for i, card_url in enumerate(sorted(card_urls), start=1):
            result = self.card_processor.process_single_card(card_url, product_type)
            all_results.append(result)

            if i < total_count and self.batch_delay > 0:
                logger.debug("Waiting %d seconds before next card...", self.batch_delay)
                time.sleep(self.batch_delay)

            if i % 10 == 0 or i == total_count:
                logger.debug("Processing progress: %d/%d (%.1f%%)",
                             i, total_count, i / total_count * 100)

        successful = sum(1 for r in all_results if r.get('success'))
        failed = total_count - successful

        logger.info("Batch processing completed for %s: %d/%d successful (%d failed)",
                    product_type, successful, total_count, failed)

        if failed > 0:
            logger.warning("%s cards had processing errors: %d failures", product_type, failed)

        return all_results