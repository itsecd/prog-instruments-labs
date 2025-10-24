from book_parser import LabirintParser
import time
import random


def main():
    """Основная функция с демонстрацией реальных результатов."""
    parser = LabirintParser()
    
    # Поисковые запросы для демонстрации работы
    search_queries = [
        "Python программирование",
        "Гарри Поттер",
        "Агата Кристи",
    ]
    
    all_books = []
    
    print("🔍 ЗАПУСК ПАРСЕРА LABIRINT.RU")
    print("=" * 60)
    print("Используются сложные регулярные выражения для:")
    print("  • Валидации URL и данных")
    print("  • Извлечения цен, рейтингов, ISBN")
    print("  • Очистки HTML контента")
    print("  • Парсинга структурированных данных")
    print("=" * 60)
    
    for query in search_queries:
        print(f"\n📖 Поиск: '{query}'")
        book_urls = parser.search_books(query, limit=2)
        
        if not book_urls:
            print(f"  ❌ Не найдено книг по запросу")
            continue
            
        for i, url in enumerate(book_urls, 1):
            print(f"  {i}. Анализ страницы...")
            book = parser.parse_book_page(url)
            if book:
                all_books.append(book)
                discount_info = f" (СКИДКА: {book.discount_price} руб)" if book.discount_price else ""
                print(f"     ✅ УСПЕХ: '{book.title}'")
                print(f"        Автор: {book.author}")
                print(f"        Цена: {book.price} руб{discount_info}")
                print(f"        Рейтинг: {book.rating}/5")
                if book.isbn:
                    print(f"        ISBN: {book.isbn}")
            else:
                print(f"     ❌ Не удалось обработать страницу")
            
            time.sleep(random.uniform(2, 4))
    
    # Сохранение и вывод результатов
    if all_books:
        filename = "real_parsing_results.csv"
        parser.save_to_csv(all_books, filename)
        parser.print_statistics(all_books)
        
        # Демонстрация реальных данных
        print(f"\n🎯 ДЕМОНСТРАЦИЯ РЕАЛЬНЫХ ДАННЫХ:")
        print("=" * 60)
        for i, book in enumerate(all_books[:5], 1):
            print(f"\n{i}. 📖 {book.title}")
            print(f"   👤 Автор: {book.author}")
            print(f"   💰 Цена: {book.price} руб" + 
                  (f" (экономия {book.price - book.discount_price} руб)" if book.discount_price else ""))
            print(f"   ⭐ Рейтинг: {book.rating}/5")
            print(f"   🏢 Издательство: {book.publisher}")
            print(f"   📅 Год: {book.year if book.year else 'Не указан'}")
            print(f"   📄 Страниц: {book.pages if book.pages else 'Не указано'}")
            if book.isbn:
                print(f"   🔢 ISBN: {book.isbn}")
        
        print(f"\n💾 Все данные сохранены в: {filename}")
        print("📈 Для анализа откройте файл в Excel или Google Sheets")
        
    else:
        print("\n❌ Парсер не смог получить данные")
        print("💡 Возможные решения:")
        print("   • Проверьте интернет-соединение")
        print("   • Обновите регулярные выражения в regex_config.py")
        print("   • Попробуйте другие поисковые запросы")


if __name__ == "__main__":
    main()
