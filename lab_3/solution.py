import re
import csv

HTML_FILE = 'test_shop.html'
CSV_FILE = 'products.csv'

SKU_PATTERN = re.compile(r'SKU:\s*([A-Z]{2}-\d{5})')
SKU_RAW_PATTERN = re.compile(r'SKU:\s*(\S+)')
TITLE_PATTERN = re.compile(r'<h2 class="title">(.*?)</h2>')
PRICE_PATTERN = re.compile(r'class="price">([\d\s,.]+(?:[₽$]|руб\.?))')
OLD_PRICE_PATTERN = re.compile(r'class="old-price">([\d\s,.]+(?:[₽$]|руб\.?))')
RATING_PATTERN = re.compile(r'([★]+)[☆]*')
REVIEWS_PATTERN = re.compile(r'\((\d+)\s+(?:отзыв[а-я]*|reviews?)\)')
PRODUCT_PATTERN = re.compile(r'<div class="product">(.*?)</div>', re.DOTALL)
CLEAN_PRICE_PATTERN = re.compile(r'[^\d.,]')


def generate_test_html():
    return """<!DOCTYPE html>
<html>
<head><title>Test Shop</title></head>
<body>
<div class="product">
  <h2 class="title">Ноутбук ASUS VivoBook</h2>
  <span class="price">49 990 ₽</span>
  <span class="old-price">59 990 ₽</span>
  <div class="rating">★★★★☆ <span class="reviews">(128 отзывов)</span></div>
  <span class="sku">SKU: NB-28471</span>
</div>
<div class="product">
  <h2 class="title">Смартфон Samsung Galaxy</h2>
  <span class="price">32 500руб</span>
  <div class="rating">★★★★★ <span class="reviews">(542 отзыва)</span></div>
  <span class="sku">SKU: SM-99312</span>
</div>
<div class="product">
  <h2 class="title">Наушники Sony WH-1000XM4</h2>
  <span class="price">$299.99</span>
  <div class="rating">★★★☆☆ <span class="reviews">(89 reviews)</span></div>
  <span class="sku">SKU: SN-44100</span>
</div>
<div class="product">
  <h2 class="title">Клавиатура Logitech MX Keys</h2>
  <span class="price">12 499 руб.</span>
  <div class="rating">★★★★☆ <span class="reviews">(215 отзывов)</span></div>
  <span class="sku">SKU: LG-77201</span>
</div>
<div class="product">
  <h2 class="title">Товар без цены</h2>
  <div class="rating">★★★★★ <span class="reviews">(10 отзывов)</span></div>
  <span class="sku">SKU: XX-00000</span>
</div>
<div class="product">
  <h2 class="title">Товар без SKU</h2>
  <span class="price">1 000 ₽</span>
  <div class="rating">★★☆☆☆ <span class="reviews">(5 отзывов)</span></div>
</div>
<div class="product">
  <h2 class="title">Битый SKU</h2>
  <span class="price">500 ₽</span>
  <div class="rating">★★★★☆ <span class="reviews">(30 отзывов)</span></div>
  <span class="sku">SKU: bad_sku</span>
</div>
</body>
</html>"""


def save_html(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def load_html(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_title(block):
    match = TITLE_PATTERN.search(block)
    return match.group(1).strip() if match else None


def extract_price(block, pattern):
    match = pattern.search(block)
    return match.group(1).strip() if match else None


def extract_rating(block):
    match = RATING_PATTERN.search(block)
    return len(match.group(1)) if match else 0


def extract_reviews(block):
    match = REVIEWS_PATTERN.search(block)
    return int(match.group(1)) if match else 0


def extract_sku(block):
    valid_match = SKU_PATTERN.search(block)
    raw_match = SKU_RAW_PATTERN.search(block)
    return valid_match.group(1) if valid_match else None, raw_match.group(1) if raw_match else None


def normalize_price(raw):
    if not raw:
        return None
    cleaned = CLEAN_PRICE_PATTERN.sub('', raw).replace(' ', '').replace(',', '.')
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_product(block):
    title = extract_title(block)
    price_raw = extract_price(block, PRICE_PATTERN)
    old_price_raw = extract_price(block, OLD_PRICE_PATTERN)
    rating = extract_rating(block)
    reviews = extract_reviews(block)
    sku, sku_raw = extract_sku(block)
    return {
        'title': title,
        'price_raw': price_raw,
        'price': normalize_price(price_raw),
        'old_price_raw': old_price_raw,
        'old_price': normalize_price(old_price_raw),
        'rating': rating,
        'reviews': reviews,
        'sku': sku,
        'sku_valid': bool(sku),
        'sku_invalid_present': bool(sku_raw) and not sku,
    }


def save_to_csv(products, path):
    fieldnames = [
        'title', 'price_raw', 'price', 'old_price_raw', 'old_price',
        'rating', 'reviews', 'sku', 'sku_valid'
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for product in products:
            writer.writerow({k: product[k] for k in fieldnames})


def print_statistics(products):
    prices = [p['price'] for p in products if p['price'] is not None]
    valid_skus = sum(1 for p in products if p['sku_valid'])
    invalid_skus = sum(1 for p in products if p['sku_invalid_present'])
    missing_fields = sum(1 for p in products if not p['price'] or not p['sku'])

    print(f"Всего товаров: {len(products)}")
    print(f"Валидных SKU: {valid_skus}")
    print(f"Невалидных SKU: {invalid_skus}")
    print(f"Товаров с пропущенными полями: {missing_fields}")
    if prices:
        print(f"Средняя цена: {sum(prices) / len(prices):.2f}")
        print(f"Макс. цена: {max(prices):.2f}")
        print(f"Мин. цена: {min(prices):.2f}")
    else:
        print("Цен нет")


def main():
    save_html(HTML_FILE, generate_test_html())
    html = load_html(HTML_FILE)
    blocks = PRODUCT_PATTERN.findall(html)
    products = [parse_product(block) for block in blocks]
    save_to_csv(products, CSV_FILE)
    print_statistics(products)
    print(f"\nCSV сохранён: {CSV_FILE}")


if __name__ == '__main__':
    main()