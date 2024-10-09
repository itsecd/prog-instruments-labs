"""
This module defines the `Questions` class, which is used to manage a queue of questions
for studying or testing. The class supports operations for learning new questions,
repeating old questions, tracking learned and unlearned questions, and handling
randomized or sequential question selection.
"""

import random
from typing import List, Union

class Questions:
    """
    The `Questions` class manages a queue of questions for learning and review.
    
    It supports the following features:
    - Loading questions from files, categorized as 'new', 'old', or 'all'.
    - Shuffling the order of questions if `random_state` is enabled.
    - Tracking progress of learning, including counts of learned, unlearned, and in-process questions.
    - Adding questions to the 'learned' list after they are answered correctly multiple times.
    - Removing questions from the unlearned list after successful learning.
    - Handling failed questions, skipping questions, and managing the question queue dynamically.
    """
    files = {
        "all": "src/questions.txt",
        "old": "src/learned_questions.txt",
        "new": "src/unlearned_questions.txt",
    }

    status = None  # Can be 'old', 'new', or 'all'
    queue: List[List[Union[int, str]]] = []
    learned_count: int = 0
    inprocess_count: int = 0
    random_state: bool = True

    @classmethod
    def clear_data(cls) -> None:
        """Resets the question queue and counters."""
        cls.queue = []
        cls.learned_count = 0
        cls.inprocess_count = 0

    @classmethod
    def repeat_old(cls) -> None:
        """Resets the queue and loads old questions."""
        cls.queue = []
        cls.status = "old"
        cls.read_file()

    @classmethod
    def learn_new(cls) -> None:
        """Resets the queue and loads new questions."""
        cls.queue = []
        cls.status = 'new'
        cls.read_file()

    @classmethod
    def repeat_all(cls) -> None:
        """Resets the queue and loads all questions."""
        cls.queue = []
        cls.status = 'all'
        cls.read_file()

    @classmethod
    def get_question(cls) -> Union[str, None]:
        """Returns the current question if available, else returns None."""
        if len(cls.queue) == 0:
            return None
        return cls.queue[0][0]

    @classmethod
    def read_file(cls):
        """Reads questions from the file and populates the queue."""
        with open(cls.files[cls.status], "r", encoding="utf-8") as f:
            if cls.random_state:
                for text in f.readlines():
                    cls.queue.insert(random.randint(0, len(cls.queue) + 1), [text, 0])
            else:
                for text in f.readlines():
                    cls.queue.append([text, 0])

    @classmethod
    def question_accept(cls) -> None:
        """Marks the current question as correctly answered and updates the queue."""
        cls.queue[0][1] += 1
        if cls.queue[0][1] > 3:
            cls.add_learned()
            text = cls.queue.pop(0)[0]
            cls.remove_from_unlearned(text)
            cls.learned_count += 1
            if cls.inprocess_count >= 1:
                cls.inprocess_count -= 1
        else:
            if cls.queue[0][1] == 1:
                cls.inprocess_count += 1
            place: int = cls.queue[0][1] * 4
            text = cls.queue[0]
            cls.queue.pop(0)
            cls.queue.insert(place, text)

    @classmethod
    def update_learned(cls):
        """Updates the list of learned questions in the file."""
        with open(cls.files["old"], "w", encoding="utf-8") as f:
            for elem in cls.queue:
                f.write(elem[0])

    @classmethod
    def add_learned(cls):
        """Adds the current question to the list of learned questions."""
        if cls.status == "new":
            with open(cls.files["old"], "a+", encoding="utf-8") as f:
                f.write(cls.queue[0][0])
        elif cls.status == 'all':
            learned = set()
            with open(cls.files['old'], 'r', encoding='utf-8') as f:
                for text in f.readlines():
                    learned.add(text)
            learned.add(cls.queue[0][0])
            with open(cls.files['old'], 'w', encoding='utf-8') as f:
                for text in learned:
                    f.write(text)

    @classmethod
    def remove_from_unlearned(cls, unlearned_text: str) -> None:
        """Removes the learned question from the unlearned questions file."""
        if cls.status == "new":
            with open(cls.files["new"], "w", encoding="utf-8") as f:
                for elem in cls.queue:
                    f.write(elem[0])
        elif cls.status == 'all':
            unlearned = set()
            with open(cls.files['new'], 'r', encoding='utf-8') as f:
                for text in f.readlines():
                    if unlearned_text != text:
                        unlearned.add(text)
            with open(cls.files['new'], 'w', encoding='utf-8') as f:
                for text in unlearned:
                    f.write(text)

    @classmethod
    def delete_from_learned(cls, unlearned_text: str):
        """Removes the failed question from the learned questions file."""
        if cls.status == "old":
            with open(cls.files["old"], "w", encoding="utf-8") as f:
                for text in cls.queue[1:]:
                    f.write(text[0])
        elif cls.status == 'all':
            learned = set()
            with open(cls.files['old'], 'r', encoding='utf-8') as f:
                for text in f.readlines():
                    if unlearned_text != text:
                        learned.add(text)
            with open(cls.files['old'], 'w', encoding='utf-8') as f:
                for text in learned:
                    f.write(text)

    @classmethod
    def add_to_unlearned(cls, unlearned_text: str) -> None:
        """Adds a failed question back to the unlearned list."""
        if cls.status == "old":
            with open(cls.files["new"], "a+", encoding="utf-8") as f:
                f.write(unlearned_text)
        elif cls.status == 'all':
            unlearned = set()
            with open(cls.files['new'], 'r', encoding='utf-8') as f:
                for text in f.readlines():
                    unlearned.add(text)
            unlearned.add(unlearned_text)
            with open(cls.files['new'], 'w', encoding='utf-8') as f:
                for text in unlearned:
                    f.write(text)

    @classmethod
    def question_failed(cls) -> None:
        """Handles the case when the user fails a question."""
        if cls.inprocess_count >= 1 and cls.queue[0][1] > 0:
            cls.inprocess_count -= 1
        cls.queue[0][1] = 0
        cls.delete_from_learned(cls.queue[0][0])
        cls.add_to_unlearned(cls.queue[0][0])

    @classmethod
    def get_learned(cls) -> int:
        """Returns the count of learned questions."""
        return cls.learned_count

    @classmethod
    def get_unlearned(cls) -> int:
        """Returns the number of unlearned questions in the queue."""
        return len(cls.queue)

    @classmethod
    def get_inprocess(cls) -> int:
        """Returns the count of questions currently being processed."""
        return cls.inprocess_count

    @classmethod
    def skip_question(cls) -> None:
        """Skips the current question and moves it to the end of the queue."""
        text = cls.queue.pop(0)
        if text[1] > 0:
            cls.inprocess_count -= 1
        cls.queue.append([text[0], 0])

    @classmethod
    def set_random_state(cls, random_state: bool) -> None:
        """Sets the randomization state for the queue."""
        cls.random_state = random_state

