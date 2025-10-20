import re
import csv
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, quote
import time
import random


@dataclass
class Book:
    """Класс для хранения информации о книге."""
    title: str
    author: str
    price: float
    discount_price: Optional[float]
    rating: float
    url: str
    isbn: Optional[str]
    publisher: str
    year: int
    pages: int


class LabirintParser:
    """Парсер для сайта Labirint.ru с использованием сложных регулярных выражений."""
    
    def __init__(self):
        self.base_url = "https://www.labirint.ru"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Сложные регулярные выражения для валидации и парсинга
        self.regex_patterns = {
            # 1. Валидация URL Labirint (сложное)
            'url_validator': re.compile(
                r'^https?://(?:www\.)?labirint\.ru/(?:books|games|office|souvenirs|multimedia)/\d+/?\??(?:.*#.*)?$'
            ),
            
            # 2. Извлечение ID товара из URL (сложное)
            'item_id_extractor': re.compile(
                r'/(?:books|games|office|souvenirs|multimedia)/(\d+)(?:/|%|$|\?|#)'
            ),
            
            # 3. Валидация ISBN-10 и ISBN-13 (очень сложное)
            'isbn_validator': re.compile(
                r'^(?:(?:ISBN(?:-1[03])?:?\s*)?(?=[-0-9\sX]{10,17}(?:\s|$))(?:\d[-0-9\s]{9,}[\dX]|\d{1,5}[-0-9\s]+[-0-9\s]+[\dX]))$'
            ),
            
            # 4. Извлечение цен с учетом форматов и валют (сложное)
            'price_extractor': re.compile(
                r'(?:\b|>)(\d{1,3}(?:\s?\d{3})*(?:[.,]\d{2})?)(?:\s*(?:руб|р|₽|rub)|<|$)'
            ),
            
            # 5. Валидация года издания с историческим контекстом (сложное)
            'year_validator': re.compile(
                r'^(1[6-9]\d{2}|20[0-2]\d|202[0-4])(?:\s*г(?:од)?\.?)?$'
            ),
            
            # 6. Извлечение рейтинга из различных форматов (сложное)
            'rating_extractor': re.compile(
                r'(?:рейтинг|rating|оценка)[^:\d]*(?:[:>]\s*)?(\d[,.]\d|\d)(?:\s*(?:из|out of|\/)\s*[5])?'
            ),
            
            # 7. Валидация имен авторов с поддержкой Unicode (сложное)
            'author_validator': re.compile(
                r'^(?!(?i:автор|author|unknown|неизвестен)\b)[A-Za-zА-Яа-яЁё\s\-\'\.\,&\(\)]{2,50}$'
            ),
            
            # 8. Извлечение количества страниц (сложное)
            'pages_extractor': re.compile(
                r'(?:страниц|pages|стр\.?)[^:\d]*(?:[:>]\s*)?(\d{1,4})(?:\s*(?:с\.|стр|pages?))?'
            ),
            
            # 9. Валидация названий издательств (сложное)
            'publisher_validator': re.compile(
                r'^(?!(?i:издательство|publisher|unknown|неизвестно)\b)[A-Za-zА-Яа-яЁё0-9\s\"\-\.,&\(\):]{2,80}$'
            ),
            
            # 10. Очистка HTML с сохранением текста (сложное)
            'html_cleaner': re.compile(
                r'<script[^>]*>.*?</script>|<style[^>]*>.*?</style>|<!--.*?-->|<[^>]+>|&(?:nbsp|lt|gt|quot|amp|#\d+);'
            )
        }
    
    def search_books(self, query: str, limit: int = 5) -> List[str]:
        """Поиск книг по запросу и получение реальных URL."""
        search_url = f"{self.base_url}/search/{quote(query)}/"
        
        try:
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            # Сложное регулярное выражение для извлечения URL книг
            book_urls_pattern = re.compile(
                r'href="(/books/\d+/[^"?]*(?:\?[^"]*)?)"[^>]*class="[^"]*product-title[^"]*"',
                re.IGNORECASE | re.DOTALL
            )
            
            matches = book_urls_pattern.findall(response.text)
            full_urls = [urljoin(self.base_url, match) for match in matches[:limit]]
            
            return list(set(full_urls))  # Убираем дубликаты
            
        except requests.RequestException as e:
            print(f"Ошибка поиска '{query}': {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Очистка текста от HTML тегов и нормализация."""
        if not text:
            return ""
        
        # Удаляем HTML теги и entities
        cleaned = self.regex_patterns['html_cleaner'].sub(' ', text)
        # Нормализуем пробелы и убираем лишние символы
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Убираем начальные/конечные знаки препинания
        cleaned = re.sub(r'^[^\wА-Яа-я]+|[^\wА-Яа-я]+$', '', cleaned)
        
        return cleaned
    
    def validate_isbn(self, isbn: str) -> Optional[str]:
        """Валидация и нормализация ISBN."""
        if not isbn:
            return None
        
        # Очищаем от лишних символов, оставляем только цифры и X
        clean_isbn = re.sub(r'[^\dX]', '', isbn.upper())
        
        if self.regex_patterns['isbn_validator'].match(clean_isbn):
            return clean_isbn
        return None
    
    def extract_price(self, price_text: str) -> float:
        """Извлечение цены из текста с поддержкой разных форматов."""
        if not price_text:
            return 0.0
        
        match = self.regex_patterns['price_extractor'].search(price_text)
        if match:
            price_str = match.group(1).replace(' ', '').replace(',', '.')
            try:
                return float(price_str)
            except ValueError:
                pass
        return 0.0
    
    def extract_rating(self, rating_text: str) -> float:
        """Извлечение рейтинга из текста."""
        if not rating_text:
            return 0.0
        
        match = self.regex_patterns['rating_extractor'].search(
            rating_text.lower()
        )
        if match:
            rating_str = match.group(1).replace(',', '.')
            try:
                rating = float(rating_str)
                return min(max(rating, 0.0), 5.0)  # Ограничение 0-5
            except ValueError:
                pass
        return 0.0
    
    def parse_book_page(self, url: str) -> Optional[Book]:
        """Парсинг страницы книги с использованием регулярных выражений."""
        
        # Валидация URL
        if not self.regex_patterns['url_validator'].match(url):
            print(f"Неверный формат URL: {url}")
            return None
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            html_content = response.text
            
            # Извлечение данных с помощью сложных регулярных выражений
            title = self._extract_title(html_content)
            author = self._extract_author(html_content)
            price_data = self._extract_prices(html_content)
            rating = self._extract_book_rating(html_content)
            isbn = self._extract_isbn(html_content)
            publisher = self._extract_publisher(html_content)
            year = self._extract_year(html_content)
            pages = self._extract_pages(html_content)
            
            if not title:
                return None
                
            return Book(
                title=title,
                author=author,
                price=price_data['price'],
                discount_price=price_data['discount_price'],
                rating=rating,
                url=url,
                isbn=isbn,
                publisher=publisher,
                year=year,
                pages=pages
            )
            
        except requests.RequestException as e:
            print(f"Ошибка при запросе {url}: {e}")
            return None
        except Exception as e:
            print(f"Неожиданная ошибка при парсинге {url}: {e}")
            return None
    
    def _extract_title(self, html: str) -> str:
        """Извлечение названия книги с помощью регулярных выражений."""
        patterns = [
            r'<meta\s+property="og:title"\s+content="([^"]+)"',
            r'<h1[^>]*data-zone-name="title"[^>]*>(.*?)</h1>',
            r'<title>([^<|]+)',
            r'class="prodtitle"[^>]*>(.*?)</'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                title = self.clean_text(match.group(1))
                if title and len(title) > 2:
                    return title
        return ""
    
    def _extract_author(self, html: str) -> str:
        """Извлечение автора книги."""
        patterns = [
            r'Авторы?[^:>]*:[^>]*>(.*?)<',
            r'<div[^>]*class="authors"[^>]*>.*?<a[^>]*>(.*?)</a>',
            r'<meta[^>]*name="author"[^>]*content="([^"]+)"',
            r'author[^>]*>([^<]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                author = self.clean_text(match.group(1))
                if (author and 
                    self.regex_patterns['author_validator'].match(author) and
                    author.lower() not in ['автор', 'author']):
                    return author
        return "Неизвестен"
    
    def _extract_prices(self, html: str) -> Dict[str, float]:
        """Извлечение цен книги (основной и со скидкой)."""
        price_patterns = {
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
        }
        
        prices = {'price': 0.0, 'discount_price': None}
        
        for price_type, patterns in price_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    price_value = self.extract_price(match.group(1))
                    if price_value > 0:
                        prices[price_type] = price_value
                        break
        
        # Если есть цена со скидкой, но нет основной - используем скидочную как основную
        if prices['discount_price'] and not prices['price']:
            prices['price'] = prices['discount_price']
            prices['discount_price'] = None
            
        return prices
    
    def _extract_book_rating(self, html: str) -> float:
        """Извлечение рейтинга книги."""
        patterns = [
            r'Рейтинг[^>]*>([^<]+)',
            r'rating[^>]*>([^<]+)',
            r'class="rating"[^>]*>([^<]+)',
            r'data-rating="([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                rating = self.extract_rating(match.group(1))
                if rating > 0:
                    return rating
        return 0.0
    
    def _extract_isbn(self, html: str) -> Optional[str]:
        """Извлечение и валидация ISBN."""
        patterns = [
            r'ISBN[^>]*>([^<]+)',
            r'isbn[^>]*>([^<]+)',
            r'978[\d-]+|\d[\d-]+\d',
            r'ISBN[^:\d]*([\d-Xx]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                isbn_candidate = self.clean_text(match.group(1))
                validated_isbn = self.validate_isbn(isbn_candidate)
                if validated_isbn:
                    return validated_isbn
        return None
    
    def _extract_publisher(self, html: str) -> str:
        """Извлечение издательства."""
        patterns = [
            r'Издательство[^>]*>([^<]+)',
            r'Издатель[^>]*>([^<]+)',
            r'publisher[^>]*>([^<]+)',
            r'class="publisher"[^>]*>.*?<a[^>]*>([^<]+)</a>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                publisher = self.clean_text(match.group(1))
                if (publisher and 
                    self.regex_patterns['publisher_validator'].match(publisher) and
                    publisher.lower() not in ['издательство', 'publisher']):
                    return publisher
        return "Неизвестно"
    
    def _extract_year(self, html: str) -> int:
        """Извлечение года издания."""
        patterns = [
            r'Год издания[^>]*>([^<]+)',
            r'Год[^>]*>([^<]+)',
            r'year[^>]*>([^<]+)',
            r'(\b20[0-2]\d\b)',
            r'(\b19[8-9]\d\b)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                year_str = self.clean_text(match.group(1))
                if (year_str.isdigit() and 
                    self.regex_patterns['year_validator'].match(year_str)):
                    return int(year_str)
        return 0
    
    def _extract_pages(self, html: str) -> int:
        """Извлечение количества страниц."""
        patterns = [
            r'Страниц[^>]*>([^<]+)',
            r'pages[^>]*>([^<]+)',
            r'(\d+)\s*стр',
            r'страниц[^:\d]*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                pages_str = self.clean_text(match.group(1))
                if pages_str.isdigit():
                    pages = int(pages_str)
                    if 1 <= pages <= 5000:  # Реалистичный диапазон
                        return pages
        return 0
    
    def save_to_csv(self, books: List[Book], filename: str = "labirint_books.csv"):
        """Сохранение данных о книгах в CSV файл."""
        if not books:
            print("Нет данных для сохранения")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = [
                'title', 'author', 'price', 'discount_price', 'rating', 
                'isbn', 'publisher', 'year', 'pages', 'url'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for book in books:
                writer.writerow({
                    'title': book.title,
                    'author': book.author,
                    'price': book.price,
                    'discount_price': book.discount_price or '',
                    'rating': book.rating,
                    'isbn': book.isbn or '',
                    'publisher': book.publisher,
                    'year': book.year,
                    'pages': book.pages,
                    'url': book.url
                })
        
        print(f" Данные сохранены в файл: {filename}")
