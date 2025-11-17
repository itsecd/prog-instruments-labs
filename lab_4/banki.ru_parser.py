import json
import time
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

BANK_CONFIGS = [
    {
        "url": "https://www.banki.ru/products/debitcards/alfabank/",
        "name": "Альфа-Банк",
        "product_type": "debitcards"
    },
    {
        "url": "https://www.banki.ru/products/creditcards/alfabank/",
        "name": "Альфа-Банк",
        "product_type": "creditcards"
    },
    {
        "url": "https://www.banki.ru/products/creditcards/sovcombank/",
        "name": "Совкомбанк",
        "product_type": "creditcards"
    },
    {
        "url": "https://www.banki.ru/products/debitcards/sovcombank/",
        "name": "Совкомбанк",
        "product_type": "debitcards"
    },
    {
        "url": "https://www.banki.ru/products/debitcards/tcs/",
        "name": "Т-Банк",
        "product_type": "debitcards"
    },
    {
        "url": "https://www.banki.ru/products/creditcards/tcs/",
        "name": "Т-Банк",
        "product_type": "creditcards"
    },
    {
        "url": "https://www.banki.ru/products/debitcards/vtb/",
        "name": "ВТБ",
        "product_type": "debitcards"
    },
    {
        "url": "https://www.banki.ru/products/creditcards/vtb/",
        "name": "ВТБ",
        "product_type": "creditcards"
    },
    {
        "url": "https://www.banki.ru/products/debitcards/gazprombank/",
        "name": "Газпромбанк",
        "product_type": "debitcards"
    },
    {
        "url": "https://www.banki.ru/products/creditcards/gazprombank/",
        "name": "Газпромбанк",
        "product_type": "creditcards"
    },
    {
        "url": "https://www.banki.ru/products/debitcards/rshb/",
        "name": "Россельхозбанк",
        "product_type": "debitcards"
    },
    {
        "url": "https://www.banki.ru/products/creditcards/rshb/",
        "name": "Россельхозбанк",
        "product_type": "creditcards"
    },
    {
        "url": "https://www.banki.ru/products/debitcards/domrfbank/",
        "name": "Банк ДОМ.РФ",
        "product_type": "debitcards"
    },
    {
        "url": "https://www.banki.ru/products/creditcards/domrfbank/",
        "name": "Банк ДОМ.РФ",
        "product_type": "creditcards"
    },
]

# Паттерны URL для разных типов карт
PRODUCT_PATTERNS = {
    "debitcards": "https://www.banki.ru/products/debitcards/card/",
    "creditcards": "https://www.banki.ru/products/creditcards/card/"
}


def parse_json_ld_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    json_ld_scripts = soup.find_all('script', type='application/ld+json')

    all_json_ld_data = []
    for script in json_ld_scripts:
        try:
            json_text = script.string.strip() if script.string else ''
            json_data = json.loads(json_text)
            all_json_ld_data.append(json_data)
        except Exception as e:
            print(f"Ошибка при обработке блока JSON-LD: {e}")
    return all_json_ld_data


def extract_bank_card_urls(json_ld_data, bank_name, product_type):
    """
    Универсальная функция для извлечения URL карт разных типов
    """
    bank_urls = set()
    url_pattern = PRODUCT_PATTERNS.get(product_type)

    if not url_pattern:
        print(f"Неизвестный тип продукта: {product_type}")
        return bank_urls

    for data in json_ld_data:
        # Обрабатываем разные типы продуктов в JSON-LD
        if isinstance(data, dict) and data.get('@type') in ['Product', 'FinancialProduct']:
            # Вариант 1: AggregateOffer с offers
            if ('offers' in data and
                    isinstance(data['offers'], dict) and
                    'offers' in data['offers']):

                for offer in data['offers']['offers']:
                    if (isinstance(offer, dict) and
                            offer.get('url', '').startswith(url_pattern) and
                            offer.get('provider', {}).get('name') == bank_name):
                        bank_urls.add(offer['url'])

            # Вариант 2: Прямой массив offers
            elif 'offers' in data and isinstance(data['offers'], list):
                for offer in data['offers']:
                    if (isinstance(offer, dict) and
                            offer.get('url', '').startswith(url_pattern) and
                            offer.get('provider', {}).get('name') == bank_name):
                        bank_urls.add(offer['url'])

            # Вариант 3: Единое предложение (для кредитных карт)
            elif ('offers' in data and
                  isinstance(data['offers'], dict) and
                  data['offers'].get('url', '').startswith(url_pattern) and
                  data['offers'].get('provider', {}).get('name') == bank_name):
                bank_urls.add(data['offers']['url'])

    return bank_urls


def extract_clean_card_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Ищем data-module-options (пробуем разные варианты)
    all_data_divs = soup.find_all('div', attrs={'data-module-options': True})

    # Пробуем найти подходящий data-module-options
    data_div = None
    if len(all_data_divs) >= 3:
        data_div = all_data_divs[2]  # Третий по счету
    elif len(all_data_divs) >= 1:
        data_div = all_data_divs[0]  # Первый, если меньше трех
    else:
        return None

    # Парсим JSON
    raw_json = data_div['data-module-options']
    try:
        card_json = json.loads(raw_json)
    except json.JSONDecodeError:
        try:
            json_text = raw_json.replace('&quot;', '"')
            card_json = json.loads(json_text)
        except Exception as e:
            print(f"Ошибка парсинга data-module-options: {e}")
            return None

    return card_json


def clean_card_data(raw_data, product_type):
    """Универсальная очистка данных для разных типов карт"""
    if not raw_data:
        return None

    data = raw_data.get('data', {})

    # Убираем рекламные предложения вкладов и другие ненужные данные
    promotional_fields = ['promo_deposit_offers', 'promo_offers', 'special_offers', 'advertising_blocks']
    for field in promotional_fields:
        if field in data:
            del data[field]

    # Создаем словарь для быстрого доступа к вкладкам
    tabs_dict = {}
    for tab in data.get('tabsContent', []):
        tab_code = tab.get('code')
        tabs_dict[tab_code] = tab

    # Бонусы (сохраняем оригинальную структуру)
    bonuses_clean = tabs_dict.get('bonuses')

    # Тарифы
    tariffs_clean = {}
    tariffs_data = tabs_dict.get('tariffs')
    if tariffs_data:
        for tariff_block in tariffs_data.get('items', []):
            block_name = tariff_block.get('label')
            tariffs_clean[block_name] = []
            for item in tariff_block.get('items', []):
                tariffs_clean[block_name].append({
                    'label': item.get('label'),
                    'value': item.get('value'),
                    'comment': item.get('comment')
                })

    # Требования
    requirements_clean = {}
    requirements_data = tabs_dict.get('req_documents')
    if requirements_data:
        for item in requirements_data.get('items', []):
            requirements_clean[item.get('label')] = item.get('value')

    # Экспертиза
    expertise_clean = {}
    expertise_data = tabs_dict.get('expertise')
    if expertise_data:
        for item in expertise_data.get('items', []):
            expertise_clean[item.get('label')] = item.get('value')

    # Базовые поля для всех типов карт
    clean_result = {
        'id': data.get('id'),
        'name': data.get('cardName'),
        'bank': data.get('bankName'),
        'rating': data.get('rating'),
        'features': data.get('featuresList', []),
        'bonuses': bonuses_clean,
        'tariffs': tariffs_clean,
        'requirements': requirements_clean,
        'expertise': expertise_clean,
        'updated_at': data.get('updatedAt'),
        'product_type': product_type
    }

    # Дополнительные поля для кредитных карт
    if product_type == 'creditcards':
        clean_result.update({
            'credit_limit': data.get('creditLimit'),
            'interest_rate': data.get('interestRate'),
            'grace_period': data.get('gracePeriod')
        })

    return clean_result


# Универсальная обработка одной карты
def process_single_card(card_url, product_type, delay=1):
    try:
        print(f"Обрабатываем карту: {card_url}")
        time.sleep(delay)
        response = cffi_requests.get(card_url, impersonate="safari15_5")
        response.raise_for_status()
        html_content = response.text

        raw_data = extract_clean_card_data(html_content)

        if not raw_data:
            return {'url': card_url, 'success': False, 'error': 'no data'}

        clean_data = clean_card_data(raw_data, product_type)

        if not clean_data:
            return {'url': card_url, 'success': False, 'error': 'failed to clean data'}

        result = {
            'url': card_url,
            **clean_data,
            'success': True
        }

        print(f"Успешно: {clean_data['name']}")
        return result

    except Exception as e:
        print(f"Ошибка при обработке {card_url}: {e}")
        return {'url': card_url, 'success': False, 'error': str(e)}


# Обработка всех карт банка
def process_all_cards(card_urls, product_type, delay=2):
    all_results = []
    total_count = len(card_urls)
    print(f"\nНачинаем обработку {total_count} карт ({product_type})...\n")

    for i, card_url in enumerate(sorted(card_urls), start=1):
        result = process_single_card(card_url, product_type, delay)
        all_results.append(result)
        print(f"Прогресс: {i}/{total_count} ({i / total_count * 100:.1f}%)")

    successful = sum(1 for r in all_results if r.get('success'))
    print(f"\nСтатистика для {product_type}:")
    print(f"Успешно: {successful}")
    print(f"Ошибки: {total_count - successful}\n")

    return all_results


# Сохранение в JSON
def save_results_to_file(results, filename="cards_data.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Данные сохранены в файл: {filename}")
    except Exception as e:
        print(f"Ошибка при сохранении: {e}")


# Статистика и итоги
def print_final_statistics(all_cards_data):
    """Выводит итоговую статистику по всем собранным данным"""
    total_cards = len(all_cards_data)
    successful_cards = sum(1 for card in all_cards_data if card.get('success'))
    failed_cards = total_cards - successful_cards

    # Статистика по типам карт
    debit_cards = [card for card in all_cards_data if card.get('product_type') == 'debitcards']
    credit_cards = [card for card in all_cards_data if card.get('product_type') == 'creditcards']

    # Статистика по банкам
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

    print(f"\n{'=' * 80}")
    print(f"ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'=' * 80}")
    print(f"Всего карт собрано: {total_cards}")
    print(f"Успешно обработано: {successful_cards}")
    print(f"С ошибками: {failed_cards}")
    print(f"Дебетовых карт: {len(debit_cards)}")
    print(f"Кредитных карт: {len(credit_cards)}")

    print(f"\n🏦 СТАТИСТИКА ПО БАНКАМ:")
    for bank_name, stats in sorted(banks.items()):
        total = stats['debit'] + stats['credit']
        print(f"  {bank_name}: {total} карт (дебетовых: {stats['debit']}, кредитных: {stats['credit']})")


#Основной запуск
if __name__ == "__main__":
    # Вместо словаря для каждого банка, создаем один список для всех карт
    all_cards_data = []

    total_processed = 0
    total_successful = 0

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
            html_content = response.text

            json_ld_data = parse_json_ld_from_html(html_content)
            card_urls = extract_bank_card_urls(json_ld_data, bank_name, product_type)

            print(f"Найдено {len(card_urls)} карт {bank_name} ({product_type}):")
            for url in sorted(card_urls):
                print(f"  - {url}")

            if card_urls:
                bank_results = process_all_cards(card_urls, product_type)

                # Добавляем все карты в общий список
                all_cards_data.extend(bank_results)

                # Статистика по текущему банку
                successful_in_bank = sum(1 for r in bank_results if r.get('success'))
                total_processed += len(bank_results)
                total_successful += successful_in_bank

                print(f"{bank_name} ({product_type}): {successful_in_bank}/{len(bank_results)} успешно")

        except Exception as e:
            print(f"Ошибка при обработке банка {bank_name} ({product_type}): {e}")
            print(f"{'=' * 60}")

    # Сохраняем ВСЕ данные в один файл
    if all_cards_data:
        # Основной файл со всеми данными
        save_results_to_file(all_cards_data, "all_cards_combined.json")

        # Дополнительно: разделяем по типам карт для удобства
        debit_cards = [card for card in all_cards_data if card.get('product_type') == 'debitcards']
        credit_cards = [card for card in all_cards_data if card.get('product_type') == 'creditcards']

        if debit_cards:
            save_results_to_file(debit_cards, "debit_cards.json")
        if credit_cards:
            save_results_to_file(credit_cards, "credit_cards.json")

        # Выводим итоговую статистику
        print_final_statistics(all_cards_data)

        print(f"\nОБРАБОТКА ЗАВЕРШЕНА!")
        print(f"Основной файл: all_cards_combined.json ({len(all_cards_data)} карт)")
        print(f"Дебетовые карты: debit_cards.json ({len(debit_cards)} карт)")
        print(f"Кредитные карты: credit_cards.json ({len(credit_cards)} карт)")
    else:
        print("\nНе удалось собрать данные ни по одной карте")

# Экспортируем функции для импорта
__all__ = [
    'BANK_CONFIGS',
    'parse_json_ld_from_html',
    'extract_bank_card_urls',
    'process_all_cards',
    'save_results_to_file',
    'print_final_statistics'
]