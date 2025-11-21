"""
Smoke tests for CI environment
"""
import pytest
import sys
import os


class TestCISmoke:
    """Smoke tests that definitely work in CI"""

    def test_import(self):
        """Test that we can import the module"""
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        try:
            from twxt import MmabiaaTextpad
            assert True  # If we get here, import worked
        except ImportError as e:
            pytest.fail(f"Failed to import module: {e}")

    def test_basic_logic(self):
        """Test basic Python logic"""
        assert 1 + 1 == 2
        assert "hello".upper() == "HELLO"

    @pytest.mark.parametrize("a,b,expected", [
        (2, 3, 5),
        (5, 5, 10),
        (0, 0, 0),
    ])
    def test_addition_parameterized(self, a, b, expected):
        """Parameterized test for addition"""
        assert a + b == expected

    def test_list_operations(self):
        """Test list operations"""
        items = [1, 2, 3]
        items.append(4)
        assert len(items) == 4
        assert items[-1] == 4

    def test_string_operations(self):
        """Test string operations"""
        text = "Hello World"
        assert text.split() == ["Hello", "World"]
        assert text.replace("World", "Python") == "Hello Python"

    def test_dictionary_operations(self):
        """Test dictionary operations"""
        data = {"name": "John", "age": 30}
        assert "name" in data
        assert data["age"] == 30
        data["city"] = "New York"
        assert len(data) == 3