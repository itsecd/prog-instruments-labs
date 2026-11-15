import csv
import re
from urllib.parse import urljoin

import requests


BASE_URL = "https://www.fish-ok.ru"
HOOKS_URL = (
    f"{BASE_URL}/catalog/kryuchki/"
    "kryuchki_cpinningiovye_flagman/"
)

WOBBLERS_URL = f"{BASE_URL}/catalog/voblery/voblery_popery/"

HOOK_PATTERN = re.compile(
    r'<a\s+href="'
    r'(/catalog/kryuchki/kryuchki_cpinningiovye_flagman/[0-9]+/)"'
    r'\s*>\s*<span>([^<]+)</span>\s*</a>'
    r'.*?<span\s+class="value">([^<]+)</span>',
    flags=re.DOTALL,
)

WOBBLER_PATTERN = re.compile(
    r'<a\s+href="'
    r'(/catalog/voblery/voblery_popery/[0-9]+/)"'
    r'\s*>\s*<span>([^<]+)</span>\s*</a>'
    r'.*?<span\s+class="value">([^<]+)</span>',
    flags=re.DOTALL,
)

def download_html(url: str, params: dict | None = None) -> str:
    """Скачивает HTML"""
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.text


def save_html(html: str, filename: str) -> None:
    """Сохраняет HTML"""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(html)


def parse_catalog(
    html: str,
    pattern: re.Pattern,
) -> list[tuple[str, str, str]]:
    """Извлекает  товары из каталога"""
    return [
        (
            name.strip(),
            urljoin(BASE_URL, link),
            availability.strip(),
        )
        for link, name, availability in pattern.findall(html)
    ]


def save_to_csv(
    hooks: list[tuple[str, str, str]],
    filename: str,
) -> None:
    """Сохраняет товары в CSV"""
    with open(filename, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["Название", "Ссылка", "Наличие"])
        writer.writerows(hooks)


def main() -> None:
    catalogs = [
        (HOOKS_URL, "hooks", 1, HOOK_PATTERN),
        (WOBBLERS_URL, "wobblers", 5, WOBBLER_PATTERN),
    ]

    for url, filename, max_pages, pattern in catalogs:
        products = []
        s_links = set()

        for page in range(1, max_pages + 1):
            html = download_html(url, params={"PAGEN_1": page})
            save_html(html, f"{filename}_page_{page}.html")

            page_products = parse_catalog(html, pattern)

            if not page_products:
                print(f"{filename}: на странице {page} нет товаров")
                break

            new_products = []

            for name, link, availability in page_products:
                if link not in s_links:
                    s_links.add(link)
                    new_products.append((name, link, availability))

            if not new_products:
                print(f"{filename}: новых товаров нет")
                break

            products.extend(new_products)

        if not products:
            print(f"{filename}: нечего сохранять")
            continue

        save_to_csv(products, f"{filename}.csv")
        print(f"В {filename}.csv сохранено товаров: {len(products)}")


if __name__ == "__main__":
    main()