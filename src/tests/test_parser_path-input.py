from src.input import JSONParser
from pathlib import Path

project_root = Path.cwd()
true_test_file = project_root / "cfg" / "input.json"


def test_absolute_path_input(capsys):
    print("Test: Input/absolute path input ", end="")
    test_input = project_root / "cfg" / "input.json"
    parser = JSONParser(test_input)
    assert parser.path == true_test_file


def test_absolute_str_input(capsys):
    print("Test: Input/absolute str input ", end="")
    test_input = str(project_root) + "/cfg/input.json"
    parser = JSONParser(test_input)
    assert parser.path == true_test_file


def test_relative_str_input(capsys):
    print("Test: Input/relative str input ", end="")
    test_input = "./cfg/input.json"
    parser = JSONParser(test_input)
    assert parser.path == true_test_file


def test_parse(capsys):
    print("Test: Input/parse ", end="")
    TEST_INPUT = {
        "material": "brick",
        "sampleLength": 0.2,
        "moistureUptakeCoefficient": 10.0,
        "freeSaturation": 300.0,
        "meanPoreSize": 10e-6,
        "freeParameter": 10,
        "numberofElements": 100,
        "timeStepSize": 0.01,
        "totalTime": 10,
        "Anfangsfeuchte": 40
    }

    parser = JSONParser(true_test_file)
    cfg = parser(validate=True)
    print("Parsed input is: ")
    #print(cfg)
    if cfg == TEST_INPUT:
        print("[SUCCESS]")
    else:
        print("[FAILURE]")
