#!/usr/bin/env python3
"""
Convenient test runner script for the Parametric Scarring Experiment.

This script provides a simple way to run all tests from the project root.

Usage:
    python run_tests.py

Remember to activate the virtual environment first:
    source .venv/bin/activate
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the test suite from the tests directory"""
    print("Running test suite from tests/ directory...")
    
    # Change to tests directory and run the test script
    test_script = Path(__file__).parent / "tests" / "run_all_tests.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_script)],
            cwd=Path(__file__).parent,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)