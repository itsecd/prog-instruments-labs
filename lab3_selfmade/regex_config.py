import re

# Группировка по назначению
REGEX_PATTERNS = {
    # ==================== ВАЛИДАЦИЯ ДАННЫХ ====================
    'validation': {
        # Валидация URL Labirint (сложное)
        'url_validator': re.compile(
            r'^https?://(?:www\.)?labirint\.ru/(?:books|games|office|souvenirs|multimedia)/\d+/?\??(?:.*#.*)?$'
        ),
        
        # Валидация ISBN-10 и ISBN-13 (очень сложное)
        'isbn_validator': re.compile(
            r'^(?:(?:ISBN(?:-1[03])?:?\s*)?(?=[-0-9\sX]{10,17}(?:\s|$))(?:\d[-0-9\s]{9,}[\dX]|\d{1,5}[-0-9\s]+[-0-9\s]+[\dX]))$'
        ),
        
        # Валидация года издания
        'year_validator': re.compile(
            r'^(1[6-9]\d{2}|20[0-2]\d|202[0-4])(?:\s*г(?:од)?\.?)?$'
        ),
        
        # Валидация имен авторов
        'author_validator': re.compile(
            r'^(?!(?i:автор|author|unknown|неизвестен)\b)[A-Za-zА-Яа-яЁё\s\-\'\.\,&\(\)]{2,50}$'
        ),
        
        # Валидация названий издательств
        'publisher_validator': re.compile(
            r'^(?!(?i:издательство|publisher|unknown|неизвестно)\b)[A-Za-zА-Яа-яЁё0-9\s\"\-\.,&\(\):]{2,80}$'
        ),
    },
    
    # ==================== ИЗВЛЕЧЕНИЕ ДАННЫХ ====================
    'extraction': {
        # Извлечение ID товара из URL
        'item_id_extractor': re.compile(
            r'/(?:books|games|office|souvenirs|multimedia)/(\d+)(?:/|%|$|\?|#)'
        ),
        
        # Извлечение цен с учетом форматов и валют
        'price_extractor': re.compile(
            r'(?:\b|>)(\d{1,3}(?:\s?\d{3})*(?:[.,]\d{2})?)(?:\s*(?:руб|р|₽|rub)|<|$)'
        ),
        
        # Извлечение рейтинга из различных форматов
        'rating_extractor': re.compile(
            r'(?:рейтинг|rating|оценка)[^:\d]*(?:[:>]\s*)?(\d[,.]\d|\d)(?:\s*(?:из|out of|\/)\s*[5])?'
        ),
        
        # Извлечение количества страниц
        'pages_extractor': re.compile(
            r'(?:страниц|pages|стр\.?)[^:\d]*(?:[:>]\s*)?(\d{1,4})(?:\s*(?:с\.|стр|pages?))?'
        ),
    },
    
    # ==================== ПАРСИНГ HTML ====================
    'html_parsing': {
        # Очистка HTML с сохранением текста
        'html_cleaner': re.compile(
            r'<script[^>]*>.*?</script>|<style[^>]*>.*?</style>|<!--.*?-->|<[^>]+>|&(?:nbsp|lt|gt|quot|amp|#\d+);'
        ),
        
        # Поиск URL книг в поисковой выдаче
        'book_urls_finder': re.compile(
            r'href="(/books/\d+/[^"?]*(?:\?[^"]*)?)"[^>]*class="[^"]*product-title[^"]*"',
            re.IGNORECASE | re.DOTALL
        ),
        
        # Извлечение названия книги
        'title_patterns': [
            r'<meta\s+property="og:title"\s+content="([^"]+)"',
            r'<h1[^>]*data-zone-name="title"[^>]*>(.*?)</h1>',
            r'<title>([^<|]+)',
            r'class="prodtitle"[^>]*>(.*?)</'
        ],
        
        # Извлечение автора
        'author_patterns': [
            r'Авторы?[^:>]*:[^>]*>(.*?)<',
            r'<div[^>]*class="authors"[^>]*>.*?<a[^>]*>(.*?)</a>',
            r'<meta[^>]*name="author"[^>]*content="([^"]+)"',
            r'author[^>]*>([^<]+)'
        ],
        
        # Извлечение цен из HTML
        'price_patterns': {
            'price': [
                r'class="buying-priceold-val"[^>]*>([^<]+)',
                r'Цена[^>]*>([^<]+)',
                r'price[^>]*>([^<]+)'
            ],
            'discount_price': [
                r'class="buying-pricenew-val"[^>]*>([^<]+)',
                r'Со\s+скидкой[^>]*>([^<]+)',
                r'new-price[^>]*>([^<]+)'
            ]
        },
        
        # Извлечение рейтинга из HTML
        'rating_patterns': [
            r'Рейтинг[^>]*>([^<]+)',
            r'rating[^>]*>([^<]+)',
            r'class="rating"[^>]*>([^<]+)',
            r'data-rating="([^"]+)"'
        ],
        
        # Извлечение ISBN
        'isbn_patterns': [
            r'ISBN[^>]*>([^<]+)',
            r'isbn[^>]*>([^<]+)',
            r'978[\d-]+|\d[\d-]+\d',
            r'ISBN[^:\d]*([\d-Xx]+)'
        ],
        
        # Извлечение издательства
        'publisher_patterns': [
            r'Издательство[^>]*>([^<]+)',
            r'Издатель[^>]*>([^<]+)',
            r'publisher[^>]*>([^<]+)',
            r'class="publisher"[^>]*>.*?<a[^>]*>([^<]+)</a>'
        ],
        
        # Извлечение года издания
        'year_patterns': [
            r'Год издания[^>]*>([^<]+)',
            r'Год[^>]*>([^<]+)',
            r'year[^>]*>([^<]+)',
            r'(\b20[0-2]\d\b)',
            r'(\b19[8-9]\d\b)'
        ],
        
        # Извлечение количества страниц из HTML
        'pages_patterns': [
            r'Страниц[^>]*>([^<]+)',
            r'pages[^>]*>([^<]+)',
            r'(\d+)\s*стр',
            r'страниц[^:\d]*(\d+)'
        ]
    }
}
