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

PRODUCT_PATTERNS = {
    "debitcards": "https://www.banki.ru/products/debitcards/card/",
    "creditcards": "https://www.banki.ru/products/creditcards/card/"
}

REQUEST_DELAY = 1
BATCH_DELAY = 2
USER_AGENT = "safari15_5"

PROMOTIONAL_FIELDS = [
    'promo_deposit_offers',
    'promo_offers',
    'special_offers',
    'advertising_blocks'
]

MAIN_OUTPUT_FILE = "all_cards_combined.json"
DEBIT_CARDS_FILE = "debit_cards.json"
CREDIT_CARDS_FILE = "credit_cards.json"
