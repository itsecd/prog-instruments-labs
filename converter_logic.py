"""Pure logic for currency conversion (separated for unit testing)."""


def parse_rate(soup, currency_name):
    """Extract exchange rate for a given currency from BeautifulSoup table."""
    if soup is None:
        raise ValueError("Exchange rates table is not loaded")

    table = soup.find('table', class_='tablesorter ratesTable')
    if table is None:
        raise ValueError("Rates table not found in HTML")

    cell = table.find('td', text=lambda x: x == currency_name)
    if cell is None:
        raise ValueError(f"Currency '{currency_name}' not found in table")

    rate_text = cell.find_next('td').get_text()
    try:
        return float(rate_text)
    except ValueError:
        raise ValueError(f"Invalid rate value: '{rate_text}'")


def convert_amount(amount, rate):
    """Convert amount by rate and return formatted string with 2 decimals."""
    if rate < 0:
        raise ValueError("Rate must be non-negative")
    if amount < 0:
        raise ValueError("Amount must be non-negative")
    return str('%.2f' % (float(amount) * float(rate)))