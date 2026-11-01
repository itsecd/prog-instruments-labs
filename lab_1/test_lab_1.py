import csv
import sys
from pathlib import Path

import pytest

from lab_1 import AnnotationIterator, parse_args, write_annotation 

def test_annotation_iterator() -> None:
    test_file = Path("test_annotation.csv")
    data = [
        {"absolute_path": "/img/1.jpg", "relative_path": "cat/1.jpg", "keyword": "cat"},
        {"absolute_path": "/img/2.jpg", "relative_path": "dog/2.jpg", "keyword": "dog"},
    ]
    write_annotation(str(test_file), data)

    iterator = AnnotationIterator(str(test_file))
    rows = list(iterator)
    
    assert len(rows) == 2
    assert rows[0]["keyword"] == "cat"
    assert rows[1]["keyword"] == "dog"
    
    test_file.unlink()


def test_write_annotation() -> None:
    test_file = Path("test_write.csv")
    data = [
        {"absolute_path": "/a/1.jpg", "relative_path": "1.jpg", "keyword": "cat"},
        {"absolute_path": "/b/2.jpg", "relative_path": "2.jpg", "keyword": "dog"},
    ]
    
    write_annotation(str(test_file), data)
    
    assert test_file.exists()
    
    with open(test_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 2
    assert rows[0]["keyword"] == "cat"
    assert rows[1]["keyword"] == "dog"
    
    test_file.unlink()


def test_parse_args_invalid_number() -> None:
    original_argv = sys.argv
    
    try:
        sys.argv = ["lab_1.py", "save_dir", "ann.csv", "10", "keyword"]
        
        with pytest.raises(ValueError):
            parse_args()
    finally:
        sys.argv = original_argv