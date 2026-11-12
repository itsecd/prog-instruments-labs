import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import data_loader  # noqa: E402


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(data_loader, "_BASE_DIR", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, text):
        with open(os.path.join(self.tmp.name, name), "w") as f:
            f.write(text)

    def test_hangman_words_are_lowercased(self):
        self.write("hangman_words.json", json.dumps(["Apple", "PEAR"]))
        self.assertEqual(data_loader.load_hangman_words(), ["apple", "pear"])

    def test_hangman_missing_file_returns_error_marker(self):
        self.assertEqual(data_loader.load_hangman_words(), ["error"])

    def test_hangman_rejects_non_string_items(self):
        self.write("hangman_words.json", json.dumps(["ok", 5]))
        self.assertEqual(data_loader.load_hangman_words(), ["error"])

    def test_hangman_invalid_json(self):
        self.write("hangman_words.json", "{not json")
        self.assertEqual(data_loader.load_hangman_words(), ["error"])

    def test_quiz_questions_loaded(self):
        questions = [{"question": "2+2?", "answer": "4"}]
        self.write("quiz_data.json", json.dumps(questions))
        self.assertEqual(data_loader.load_quiz_questions(), questions)

    def test_quiz_missing_or_invalid_returns_empty_list(self):
        self.assertEqual(data_loader.load_quiz_questions(), [])
        self.write("quiz_data.json", "oops")
        self.assertEqual(data_loader.load_quiz_questions(), [])


if __name__ == "__main__":
    unittest.main()