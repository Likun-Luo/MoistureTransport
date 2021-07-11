"""Tests for path parsing.

Different types of path entries are tested (absolute, relative in string or pathlib.Path form)
"""

from src.input import JSONParser
from pathlib import Path

project_root = Path.cwd()
true_test_file = project_root / "cfg" / "defaults" / "input.json"

def test_absolute_path_input(capsys):
    print("Test: Input/absolute path input ", end="")
    test_input = project_root / "cfg" / "defaults" / "input.json"
    parser = JSONParser(test_input)
    assert parser.path == true_test_file


def test_absolute_str_input(capsys):
    print("Test: Input/absolute str input ", end="")
    test_input = str(project_root) + "/cfg/defaults/input.json"
    parser = JSONParser(test_input)
    assert parser.path == true_test_file


def test_relative_str_input(capsys):
    print("Test: Input/relative str input ", end="")
    test_input = "./cfg/defaults/input.json"
    parser = JSONParser(test_input)
    assert parser.path == true_test_file
