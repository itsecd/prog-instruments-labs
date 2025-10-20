
from book_parser import LabirintParser
import time
import random


def main():
    """функция для демонстрации работы"""
    parser = LabirintParser()
    
    # поисковые запросы для получения работающих URL
    search_queries = [
        "Пушкин",
        "Толстой Война и мир", 
        "Достоевский Преступление и наказание",
        "programming python",
        "Harry Potter"
    ]
    
    all_books = []
    
    print(" Начинаем поиск и парсинг книг с Labirint.ru...")
    print("=" * 60)
    
    for query in search_queries:
        print(f"\n Ищем: '{query}'")
        book_urls = parser.search_books(query, limit=3)
        
        if not book_urls:
            print(f"  ❌ Не найдено книг по запросу '{query}'")
            continue
            
        for i, url in enumerate(book_urls, 1):
            print(f"  {i}. Парсим: {url}")
            book = parser.parse_book_page(url)
            if book:
                all_books.append(book)
                discount_info = f" (скидка: {book.discount_price} руб)" if book.discount_price else ""
                print(f"     ✅ '{book.title}' - {book.price} руб{discount_info}")
            else:
                print(f"  ❌ Не удалось распарсить книгу")
            
            # пауза между запросами
            time.sleep(random.uniform(1, 3))
    
    if all_books:
        # Сохраняем 
        filename = f"labirint_books_{len(all_books)}.csv"
        parser.save_to_csv(all_books, filename)
        
        # наша статистика
        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА ПАРСИНГА:")
        print("=" * 60)
        print(f"   Обработано книг: {len(all_books)}")
        print(f"   💰 Средняя цена: {sum(b.price for b in all_books) / len(all_books):.2f} руб")
        print(f"   ⭐ Средний рейтинг: {sum(b.rating for b in all_books) / len(all_books):.2f}/5")
        
        books_with_discount = sum(1 for b in all_books if b.discount_price)
        print(f"   Книг со скидкой: {books_with_discount}")
        
        books_with_isbn = sum(1 for b in all_books if b.isbn)
        print(f"   Книг с ISBN: {books_with_isbn}")
        
        # примеры
        print(f"\n📖 ПРИМЕРЫ НАЙДЕННЫХ КНИГ:")
        for i, book in enumerate(all_books[:3], 1):
            print(f"   {i}. {book.title}")
            print(f"      Автор: {book.author}")
            print(f"      Цена: {book.price} руб" + 
                  (f" (скидка: {book.discount_price} руб)" if book.discount_price else ""))
            print(f"      Рейтинг: {book.rating}/5")
            print()
    else:
        print("\n❌ Не удалось получить данные ни об одной книге")
        print("Возможные причины:")
        print("   - Проблемы с интернет-соединением")
        print("   - Изменения в структуре сайта Labirint.ru")
        print("   - Блокировка запросов")


if __name__ == "__main__":
    main()
