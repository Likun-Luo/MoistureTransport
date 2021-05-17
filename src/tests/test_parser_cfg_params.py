from src.input import JSONParser
from pathlib import Path

project_root = Path.cwd()
true_test_file = project_root / "cfg" / "input.json"


# TODO : Test different parameters input
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
    #print("Parsed input is: ")
    #print(cfg)
    if cfg == TEST_INPUT:
        print("[SUCCESS]")
    else:
        print("[FAILURE]")
