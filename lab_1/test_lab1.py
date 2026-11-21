import unittest

from lab1 import count_cities, extract_cities


class TestCityStats(unittest.TestCase):
    def test_extract_cities(self) -> None:
        sample = (
            "Иванов Иван\nГород: Москва\n"
            "Смирнова Анна\nГород: г. Санкт-Петербург\n"
            "Петров Пётр\nГород:   Нижний  Новгород  \n"
        )
        expected = ["Москва", "Санкт-Петербург", "Нижний Новгород"]
        self.assertEqual(extract_cities(sample), expected)

    def test_count_cities_sorting(self) -> None:
        cities = ["Казань", "Москва", "Архангельск", "Москва", "Казань"]
        expected = [("Казань", 2), ("Москва", 2), ("Архангельск", 1)]
        self.assertEqual(count_cities(cities), expected)


if __name__ == "__main__":
    unittest.main()
