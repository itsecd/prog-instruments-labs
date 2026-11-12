"""Loading of JSON data files that live next to the script."""
import json
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_json(filename, description):
    """Read a JSON file from the script's directory.

    Returns the parsed data, or None if the file is missing or invalid
    (the problem is printed, using `description` in the message).
    """
    full_path = os.path.join(_BASE_DIR, filename)

    if not os.path.exists(full_path):
        print(f"Error: {description} data file not found at {full_path}")
        return None

    try:
        with open(full_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {filename}. Check the file format.")
    except Exception as e:
        print(f"An error occurred while reading the {description} file: {e}")
    return None


def load_hangman_words(filename="hangman_words.json"):
    """Returns a list of lowercase words, or ["error"] if they cannot be loaded."""
    data = read_json(filename, "Hangman")
    if data is None:
        return ["error"]  # default list so the game doesn't crash
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return [word.lower() for word in data]
    print(f"Error: Hangman data file is not a list of strings: {filename}")
    return ["error"]


def load_quiz_questions(filename="quiz_data.json"):
    """Returns the list of quiz questions, or [] if they cannot be loaded."""
    data = read_json(filename, "Quiz")
    return [] if data is None else data