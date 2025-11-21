#!/usr/bin/env python3
"""
Script to run tests in CI environment without GUI dependencies
"""
import sys
import os
import pytest


def main():
    """Run tests for CI environment"""
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Run only the pure logic tests that don't need GUI
    test_files = [
        "tests/test_pure_logic.py",
        "tests/test_business_rules.py"
    ]

    # Run pytest
    exit_code = pytest.main([
        *test_files,
        "-v",
        "--tb=short",
        "--cov=twxt",
        "--cov-report=term-missing"
    ])

    sys.exit(exit_code)


if __name__ == "__main__":
    main()