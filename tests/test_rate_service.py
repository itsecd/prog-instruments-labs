import unittest
from rate_service import RateService


class TestRateServiceParser(unittest.TestCase):

    MOCK_HTML = """
    <html>
    <body>
        <table class="tablesorter ratesTable">
            <tr>
                <td>US Dollar</td>
                <td>0.01205</td>
                <td>82.9902</td>
            </tr>
            <tr>
                <td>Euro</td>
                <td>0.01102</td>
                <td>90.7451</td>
            </tr>
            <tr>
                <td>Invalid Currency</td>
                <td>Not a number</td>
            </tr>
        </table>
    </body>
    </html>
    """

    def test_parse_valid_html(self):
        rates = RateService._parse(self.MOCK_HTML)
        self.assertIn("US Dollar", rates)
        self.assertAlmostEqual(rates["US Dollar"], 0.01205)
        self.assertIn("Euro", rates)

    def test_parse_invalid_number_skipped(self):
        rates = RateService._parse(self.MOCK_HTML)
        self.assertNotIn("Invalid Currency", rates)

    def test_parse_missing_table_raises(self):
        with self.assertRaises(ValueError):
            RateService._parse("<html><body>No table here</body></html>")

    def test_get_rate_loads_if_empty(self):
        service = RateService()
        service.load = lambda: setattr(service, '_rates', {"Test Coin": 42.0})
        rate = service.get_rate("Test Coin")
        self.assertEqual(rate, 42.0)


if __name__ == '__main__':
    unittest.main()