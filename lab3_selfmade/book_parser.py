import csv
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, quote
import time
import random
from regex_config import REGEX_PATTERNS


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
    """Парсер для сайта Labirint.ru с использованием регулярных выражений из конфига."""
    
    def __init__(self):
        self.base_url = "https://www.labirint.ru"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Используем regex из конфигурационного файла
        self.regex = REGEX_PATTERNS
    
    def search_books(self, query: str, limit: int = 5) -> List[str]:
        """Поиск книг по запросу и получение реальных URL."""
        search_url = f"{self.base_url}/search/{quote(query)}/"
        
        try:
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            # Используем regex из конфига для поиска URL
            matches = self.regex['html_parsing']['book_urls_finder'].findall(response.text)
            full_urls = [urljoin(self.base_url, match) for match in matches[:limit]]
            
            return list(set(full_urls))
            
        except requests.RequestException as e:
            print(f"Ошибка поиска '{query}': {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Очистка текста от HTML тегов и нормализация."""
        if not text:
            return ""
        
        # Используем regex очистки из конфига
        cleaned = self.regex['html_parsing']['html_cleaner'].sub(' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r'^[^\wА-Яа-я]+|[^\wА-Яа-я]+$', '', cleaned)
        
        return cleaned
    
    def validate_isbn(self, isbn: str) -> Optional[str]:
        """Валидация ISBN с использованием regex из конфига."""
        if not isbn:
            return None
        
        clean_isbn = re.sub(r'[^\dX]', '', isbn.upper())
        if self.regex['validation']['isbn_validator'].match(clean_isbn):
            return clean_isbn
        return None
    
    def extract_price(self, price_text: str) -> float:
        """Извлечение цены с использованием regex из конфига."""
        if not price_text:
            return 0.0
        
        match = self.regex['extraction']['price_extractor'].search(price_text)
        if match:
            price_str = match.group(1).replace(' ', '').replace(',', '.')
            try:
                return float(price_str)
            except ValueError:
                pass
        return 0.0
    
    def extract_rating(self, rating_text: str) -> float:
        """Извлечение рейтинга с использованием regex из конфига."""
        if not rating_text:
            return 0.0
        
        match = self.regex['extraction']['rating_extractor'].search(rating_text.lower())
        if match:
            rating_str = match.group(1).replace(',', '.')
            try:
                rating = float(rating_str)
                return min(max(rating, 0.0), 5.0)
            except ValueError:
                pass
        return 0.0
    
    def parse_book_page(self, url: str) -> Optional[Book]:
        """Парсинг страницы книги с использованием regex из конфига."""
        
        # Валидация URL из конфига
        if not self.regex['validation']['url_validator'].match(url):
            print(f"Неверный формат URL: {url}")
            return None
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            html_content = response.text
            
            # Извлечение данных с использованием regex паттернов из конфига
            title = self._extract_with_patterns(html_content, 'title_patterns')
            author = self._extract_with_patterns(html_content, 'author_patterns', 
                                               validator='author_validator')
            price_data = self._extract_prices(html_content)
            rating = self._extract_with_patterns(html_content, 'rating_patterns', 
                                               extractor=self.extract_rating)
            isbn = self._extract_with_patterns(html_content, 'isbn_patterns',
                                             processor=self.validate_isbn)
            publisher = self._extract_with_patterns(html_content, 'publisher_patterns',
                                                  validator='publisher_validator')
            year = self._extract_with_patterns(html_content, 'year_patterns',
                                             validator='year_validator', default=0)
            pages = self._extract_with_patterns(html_content, 'pages_patterns',
                                              default=0)
            
            if not title:
                return None
                
            return Book(
                title=title,
                author=author or "Неизвестен",
                price=price_data['price'],
                discount_price=price_data['discount_price'],
                rating=rating or 0.0,
                url=url,
                isbn=isbn,
                publisher=publisher or "Неизвестно",
                year=year or 0,
                pages=pages or 0
            )
            
        except Exception as e:
            print(f"Ошибка при парсинге {url}: {e}")
            return None
    
    def _extract_with_patterns(self, html: str, pattern_key: str, 
                             validator: str = None, processor: callable = None,
                             default: any = "") -> any:
        """Универсальный метод для извлечения данных с использованием regex из конфига."""
        patterns = self.regex['html_parsing'].get(pattern_key, [])
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                value = self.clean_text(match.group(1))
                if value:
                    # Применяем валидацию если указана
                    if validator and not self.regex['validation'].get(validator, lambda x: True).match(value):
                        continue
                    
                    # Применяем обработчик если указан
                    if processor:
                        processed_value = processor(value)
                        if processed_value:
                            return processed_value
                    else:
                        # Для числовых значений пытаемся преобразовать
                        if default == 0 and value.isdigit():
                            return int(value)
                        return value
        return default
    
    def _extract_prices(self, html: str) -> Dict[str, float]:
        """Извлечение цен с использованием regex паттернов из конфига."""
        price_patterns = self.regex['html_parsing']['price_patterns']
        prices = {'price': 0.0, 'discount_price': None}
        
        for price_type, patterns in price_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    price_value = self.extract_price(match.group(1))
                    if price_value > 0:
                        prices[price_type] = price_value
                        break
        
        if prices['discount_price'] and not prices['price']:
            prices['price'] = prices['discount_price']
            prices['discount_price'] = None
            
        return prices
    
    def save_to_csv(self, books: List[Book], filename: str = "labirint_books.csv"):
        """Сохранение данных о книгах в CSV файл."""
        if not books:
            print("Нет данных для сохранения")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['title', 'author', 'price', 'discount_price', 'rating', 
                         'isbn', 'publisher', 'year', 'pages', 'url']
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
        
        print(f"📚 Данные сохранены в файл: {filename}")
    
    def print_statistics(self, books: List[Book]):
        """Вывод статистики по найденным книгам."""
        if not books:
            return
        
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ ПАРСИНГА:")
        print("=" * 60)
        print(f"📚 Всего книг: {len(books)}")
        print(f"💰 Средняя цена: {sum(b.price for b in books) / len(books):.2f} руб")
        print(f"⭐ Средний рейтинг: {sum(b.rating for b in books) / len(books):.2f}/5")
        
        books_with_discount = sum(1 for b in books if b.discount_price)
        books_with_isbn = sum(1 for b in books if b.isbn)
        modern_books = sum(1 for b in books if b.year >= 2000)
        
        print(f"🏷️  Книг со скидкой: {books_with_discount}")
        print(f"🔢 Книг с ISBN: {books_with_isbn}")
        print(f"🆕 Книг после 2000 года: {modern_books}")
        
        # Топ-3 самых дорогих книг
        expensive_books = sorted(books, key=lambda x: x.price, reverse=True)[:3]
        print(f"\n💎 САМЫЕ ДОРОГИЕ КНИГИ:")
        for i, book in enumerate(expensive_books, 1):
            discount_info = f" (скидка: {book.discount_price} руб)" if book.discount_price else ""
            print(f"   {i}. {book.title} - {book.price} руб{discount_info}")
