import json
from typing import List, Dict, Set, Optional, Any

from bs4 import BeautifulSoup

from config import PRODUCT_PATTERNS, PROMOTIONAL_FIELDS
from logging_config import logger


class JsonLdExtractor:
    """Extracts data from JSON-LD structured data"""

    def extract_from_html(self, html_content: str) -> List[Dict]:
        """Extract all JSON-LD data from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        json_ld_scripts = soup.find_all('script', type='application/ld+json')

        all_json_ld_data = []
        for script in json_ld_scripts:
            try:
                json_text = script.string.strip() if script.string else ''
                json_data = json.loads(json_text)
                all_json_ld_data.append(json_data)
            except Exception as e:
                logger.warning("JSON-LD parsing error: %s", e)
        return all_json_ld_data

    def extract_card_urls(self, json_ld_data: List[Dict], bank_name: str,
                          product_type: str) -> Set[str]:
        """Extract card URLs from JSON-LD data"""
        bank_urls = set()
        url_pattern = PRODUCT_PATTERNS.get(product_type)

        if not url_pattern:
            print(f"Неизвестный тип продукта: {product_type}")
            return bank_urls

        for data in json_ld_data:
            self._extract_urls_from_json_ld(data, bank_name, url_pattern, bank_urls)

        return bank_urls

    def _extract_urls_from_json_ld(self, data: Dict, bank_name: str,
                                   url_pattern: str, bank_urls: Set[str]):
        """Helper method to extract URLs from different JSON-LD structures"""
        if not (isinstance(data, dict) and
                data.get('@type') in ['Product', 'FinancialProduct']):
            return

        # Variant 1: AggregateOffer with offers
        if ('offers' in data and
                isinstance(data['offers'], dict) and
                'offers' in data['offers']):
            for offer in data['offers']['offers']:
                self._add_url_if_matches(offer, bank_name, url_pattern, bank_urls)

        # Variant 2: Direct offers array
        elif 'offers' in data and isinstance(data['offers'], list):
            for offer in data['offers']:
                self._add_url_if_matches(offer, bank_name, url_pattern, bank_urls)

        # Variant 3: Single offer (for credit cards)
        elif ('offers' in data and isinstance(data['offers'], dict)):
            self._add_url_if_matches(data['offers'], bank_name, url_pattern, bank_urls)

    def _add_url_if_matches(self, offer: Dict, bank_name: str,
                            url_pattern: str, bank_urls: Set[str]):
        """Add URL to set if it matches criteria"""
        if (isinstance(offer, dict) and
                offer.get('url', '').startswith(url_pattern) and
                offer.get('provider', {}).get('name') == bank_name):
            bank_urls.add(offer['url'])


class DataModuleExtractor:
    """Extracts card data from data-module-options attribute"""

    def extract(self, html_content: str) -> Optional[Dict]:
        """Extract data from data-module-options"""
        soup = BeautifulSoup(html_content, 'html.parser')
        all_data_divs = soup.find_all('div', attrs={'data-module-options': True})

        data_div = self._find_relevant_data_div(all_data_divs)
        if not data_div:
            return None

        return self._parse_json_data(data_div['data-module-options'])

    def _find_relevant_data_div(self, data_divs: List) -> Optional[Any]:
        """Find the most relevant data-module-options div"""
        if len(data_divs) >= 3:
            return data_divs[2]
        elif len(data_divs) >= 1:
            return data_divs[0]
        return None

    def _parse_json_data(self, raw_json: str) -> Optional[Dict]:
        """Parse JSON data with error handling"""
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            try:
                json_text = raw_json.replace('&quot;', '"')
                return json.loads(json_text)
            except Exception as e:
                logger.error("Failed to parse data-module-options: %s", e)
                return None


class DataCleaner:
    """Cleans and transforms raw card data"""

    def clean(self, raw_data: Dict, product_type: str) -> Optional[Dict]:
        """Clean raw card data"""
        if not raw_data:
            return None

        data = raw_data.get('data', {})
        self._remove_promotional_data(data)

        tabs_dict = self._build_tabs_dictionary(data)

        clean_result = {
            'id': data.get('id'),
            'name': data.get('cardName'),
            'bank': data.get('bankName'),
            'rating': data.get('rating'),
            'features': data.get('featuresList', []),
            'bonuses': tabs_dict.get('bonuses'),
            'tariffs': self._extract_tariffs(tabs_dict.get('tariffs')),
            'requirements': self._extract_simple_dict(tabs_dict.get('req_documents')),
            'expertise': self._extract_simple_dict(tabs_dict.get('expertise')),
            'updated_at': data.get('updatedAt'),
            'product_type': product_type
        }

        if product_type == 'creditcards':
            clean_result.update({
                'credit_limit': data.get('creditLimit'),
                'interest_rate': data.get('interestRate'),
                'grace_period': data.get('gracePeriod')
            })

        return clean_result

    def _remove_promotional_data(self, data: Dict):
        """Remove promotional fields from data"""
        for field in PROMOTIONAL_FIELDS:
            data.pop(field, None)

    def _build_tabs_dictionary(self, data: Dict) -> Dict:
        """Build dictionary for quick tab access"""
        tabs_dict = {}
        for tab in data.get('tabsContent', []):
            tab_code = tab.get('code')
            tabs_dict[tab_code] = tab
        return tabs_dict

    def _extract_tariffs(self, tariffs_data: Optional[Dict]) -> Dict:
        """Extract tariffs information"""
        tariffs_clean = {}
        if tariffs_data:
            for tariff_block in tariffs_data.get('items', []):
                block_name = tariff_block.get('label')
                tariffs_clean[block_name] = [
                    {
                        'label': item.get('label'),
                        'value': item.get('value'),
                        'comment': item.get('comment')
                    }
                    for item in tariff_block.get('items', [])
                ]
        return tariffs_clean

    def _extract_simple_dict(self, data: Optional[Dict]) -> Dict:
        """Extract simple key-value pairs from data"""
        result = {}
        if data:
            for item in data.get('items', []):
                result[item.get('label')] = item.get('value')
        return result
