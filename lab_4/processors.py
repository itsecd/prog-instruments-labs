import time
from typing import List, Dict, Any, Set

from curl_cffi import requests as cffi_requests

from config import REQUEST_DELAY, BATCH_DELAY, USER_AGENT
from extractors import DataModuleExtractor, DataCleaner


class CardProcessor:
    """Processes individual cards"""

    def __init__(self, delay: int = REQUEST_DELAY):
        self.delay = delay
        self.data_extractor = DataModuleExtractor()
        self.data_cleaner = DataCleaner()

    def process_single_card(self, card_url: str, product_type: str) -> Dict[str, Any]:
        """Process a single card"""
        try:
            print(f"Обрабатываем карту: {card_url}")
            time.sleep(self.delay)

            response = cffi_requests.get(card_url, impersonate=USER_AGENT)
            response.raise_for_status()

            raw_data = self.data_extractor.extract(response.text)
            if not raw_data:
                return self._create_error_result(card_url, 'no data')

            clean_data = self.data_cleaner.clean(raw_data, product_type)
            if not clean_data:
                return self._create_error_result(card_url, 'failed to clean data')

            result = {
                'url': card_url,
                **clean_data,
                'success': True
            }

            print(f"Успешно: {clean_data['name']}")
            return result

        except Exception as e:
            print(f"Ошибка при обработке {card_url}: {e}")
            return self._create_error_result(card_url, str(e))

    def _create_error_result(self, card_url: str, error: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'url': card_url,
            'success': False,
            'error': error
        }


class BankProcessor:
    """Processes bank data"""

    def __init__(self, json_ld_extractor):
        self.json_ld_extractor = json_ld_extractor
        self.card_processor = CardProcessor()

    def process_all_cards(self, card_urls: Set[str], product_type: str) -> List[Dict]:
        """Process multiple card URLs"""
        all_results = []
        total_count = len(card_urls)
        print(f"\nНачинаем обработку {total_count} карт ({product_type})...\n")

        for i, card_url in enumerate(sorted(card_urls), start=1):
            result = self.card_processor.process_single_card(card_url, product_type)
            all_results.append(result)
            print(f"Прогресс: {i}/{total_count} ({i / total_count * 100:.1f}%)")

        successful = sum(1 for r in all_results if r.get('success'))
        print(f"\nСтатистика для {product_type}:")
        print(f"Успешно: {successful}")
        print(f"Ошибки: {total_count - successful}\n")

        return all_results