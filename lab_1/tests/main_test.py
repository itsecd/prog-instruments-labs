import unittest

from main import parse_music_data


class MyAwesomeTests(unittest.TestCase):
    def test_always_true(self):
        self.assertTrue(...)

    def test_math_fail(self):
        self.assertFalse(0.1 + 0.2 == 0.3)

    def test_parse_music_data(self):
        res = parse_music_data("piano")
        self.assertTrue(len(res) > 0)


if __name__ == "__main__":
    unittest.main()
