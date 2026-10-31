"""Загрузка и парсинг курсов валют (вынесено из GUI согласно SRP)."""

from typing import Dict

import requests
from bs4 import BeautifulSoup


RATES_URL = "https://www.x-rates.com/table/?from=INR&amount=1"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36"
)
TABLE_CLASS = "tablesorter ratesTable"


class RateService:
    """Источник курсов: один раз загружает страницу и отдаёт курс по имени валюты."""

    def __init__(self, url: str = RATES_URL, timeout: int = 10) -> None:
        self._url = url
        self._timeout = timeout
        self._rates: Dict[str, float] = {}

    def load(self) -> None:
        """Загружает страницу и разбирает таблицу курсов в словарь {имя: курс}."""
        response = requests.get(
            self._url, headers={"User-Agent": USER_AGENT}, timeout=self._timeout
        )
        response.raise_for_status()
        self._rates = self._parse(response.text)

    def get_rate(self, currency: str) -> float:
        """Возвращает курс валюты к INR; при пустом кэше сначала загружает таблицу."""
        if not self._rates:
            self.load()
        return self._rates[currency]

    @staticmethod
    def _parse(html: str) -> Dict[str, float]:
        """Структурный разбор таблицы: имя в первой ячейке строки, курс — во второй."""
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_=TABLE_CLASS)
        if table is None:
            raise ValueError("Rate table not found on page")

        rates: Dict[str, float] = {}
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            name = cells[0].get_text(strip=True)
            try:
                rates[name] = float(cells[1].get_text(strip=True))
            except ValueError:
                continue
        return rates