"""Tests for the Parser module, specifically the SettingsSchema.

We tried to generate as many different faulty inputs as possible:
- use the wrong type of input for a field (str for a number field, etc.)
- give out of bounds input (too large numbers, etc.)

If you can think of ways the input can be faulty, try to figure out a way to detect them and then test your detection here via a unit test!

NOTE: We did not yet implement any custom Exceptions, though it would definitely be advisable moving forward!
"""
import numpy as np
from pathlib import Path
from typing import Any
from src.input import JSONParser, SettingsSchema, number

PROJECT_ROOT = Path.cwd()
TEST_FILE = PROJECT_ROOT / "cfg" / "defaults" / "input.json"

schema = SettingsSchema()
VIABLE_FIELD_TYPES = [str, int, float]

DEFAULT_CFG = {
    "material": "brick",
    "sampleLength": 0.2,
    "moistureUptakeCoefficient": 10.0,
    "freeSaturation": 300.0,
    "meanPoreSize": 1e-2,
    "freeParameter": 10,
    "Anfangsfeuchte": 10,
    "totalTime": 10,
    "numberofElements": 10,
    "averagingMethod": "linear"
}


def manipulate_param(parameters: dict, to_change: str, value: Any):
    """copies the parameters dict and changes a parameter.
    """
    assert to_change in parameters
    new = parameters.copy()
    new[to_change] = value
    return new


def generate_faulty_types(parameters: dict, key: str, test: bool = False):
    correct_types = []

    if isinstance(schema[key][0], list) or isinstance(schema[key][0], tuple):
        correct_types += schema[key][0]
    else:
        correct_types.append(schema[key][0])
    correct_range = schema[key][1]

    incorrect_types = [t for t in VIABLE_FIELD_TYPES if t not in correct_types]
    if test:
        return correct_types, incorrect_types
    return incorrect_types


def generate_faulty_range_input(parameters,
                                key: str,
                                value: Any = None,
                                test: bool = False):
    ftype = schema[key][0]
    frange = schema[key][1]
    if ftype == number or ftype in number:
        eps = np.finfo(float).eps
        below = frange[0] * (1 - 2 * eps)
        above = frange[1] * (1 + 2 * eps)
        if test:
            return frange, (below, above)
        return below, above
    if ftype == str:
        if value == None:
            faulty = "WaterIsWet"
            # "wet":covered or saturated with water or another liquid
            # Water isn't wet by itself, but it makes other materials wet
            # when it sticks to the surface of them.
            if test:
                return frange, faulty
            return faulty
        if isinstance(value, str):
            if test:
                return frange, value
            return value
        raise ValueError(
            f"{value} is not a valid value (should be of type str)!")
    raise ValueError(f"{ftype} not one of {VIABLE_FIELD_TYPES}!")


def demo_generator_outputs():
    if 'display' not in locals():
        from IPython.core.display import display  # type: ignore
    print("Testing generate_faulty_types(...):")
    display(generate_faulty_types(1, "material", test=True))
    display(generate_faulty_types(1, "sampleLength", test=True))
    display(generate_faulty_types(1, "numberofElements", test=True))
    print("Testing generate_faulty_range_input(...):")
    display(generate_faulty_range_input(1, "material", test=True))
    display(generate_faulty_range_input(1, "sampleLength", test=True))
    display(generate_faulty_range_input(1, "numberofElements", test=True))


def test_with_default_params(capsys):
    """tests parser with (correct) default input.
    """
    print(f"Test: default input (file: {TEST_FILE})", end="")

    parser = JSONParser(TEST_FILE)
    cfg = parser(validate=True)
    #print("Parsed input is: ")
    #print(cfg)
    assert cfg == DEFAULT_CFG
    print("[SUCCESS]")


def test_with_faulty_parameter_types(capsys):
    """tests parser with (correct) default input.
    """
    print("Test: faulty inputs ", end="")

    parser = JSONParser(TEST_FILE)
    correct_cfg = parser(validate=True)

    # test types
    for key in correct_cfg:
        faulty_types = generate_faulty_types(correct_cfg, key)
        for ftype in faulty_types:
            if ftype == int:
                test_value = 1
            if ftype == float:
                test_value = 11.1
            if ftype == number:
                test_value = 99.9
            if ftype == str:
                test_value = "WaterIsWet"
            faulty_cfg = manipulate_param(correct_cfg, key, test_value)
            # with capsys.disabled():
            #     print(key, faulty_cfg)
            try:
                JSONParser.validateCFG(faulty_cfg)
                # We expect a TypeError to be raised
                raise RuntimeError(
                    f"No TypeErrors were raised --> Validation should've detected errors!"
                )
            except TypeError as e:
                # We check for type errors --> raised TypeError == Okay!
                # This is the expected behavior.
                # with capsys.disabled():
                #     print(e)
                pass
            except e:
                raise ValueError(
                    f"Error '{e}' shouldn't appear here --> only TypeErrors are acceptable here!"
                )

    print("[SUCCESS]")
