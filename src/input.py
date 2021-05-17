#!/usr/bin/env python3
"""Module for input parsing of a Simulation configuration file.

"""
import json
import pathlib
import yaml
from numpy import inf
from dataclasses import dataclass, field, asdict
from typing import Union

number = [int, float]


@dataclass
class SettingsSchema:
    # base material and geometry
    material: tuple = (str, ["brick", "cement", "wood"])
    sampleLength: tuple = (number, [1e-2, 1])  # in m, min=1cm
    moistureUptakeCoefficient: tuple = (number, [1e-1, 1000])
    freeSaturation: tuple = (number, [1, 1e6])
    meanPoreSize: tuple = (number, [1e-9, 1e-1])
    freeParameter: tuple = (number, [-inf, inf])
    # boundary & initial conditions
    Anfangsfeuchte: tuple = (number, [1e-2, 1000])
    # simulation parameters
    numberofElements: tuple = (int, [1, 1e3])
    timeStepSize: tuple = (number, [1e-6, 1])
    totalTime: tuple = (number, [1e-1, 1 * 24 * 3600])  # seconds, 3600s = 1h
    # number of settings
    NUM_PARAMETERS: tuple = (int, [15, 15])

    def __iter__(self):
        dicte = asdict(self)
        #print(dicte)
        return iter(dicte)

    def __getitem__(self, item):
        return self.__getattribute__(item)

@dataclass
class BaseParser:
    """BaseParser.

    describe
    """
    path: Union[str, pathlib.Path]

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = pathlib.Path(self.path).absolute()

    def __call__(self, validate=False):
        if validate:
            #print("Validating input...")
            return self.validateCFG(self.parse())
        #print("Input validation is turned off.")
        return self.parse()

    def parse(self):
        raise NotImplementedError("parse method not implemented!")

    def validateCFG(self, cfg):
        """validates a configuratione.

        Performs various checks on the cfg-object (python dictionary) based on the serialized json-file. Mainly, this involves checking whether the input is of valid format, size and also performs out of bounds checks.

        TODO: Implement validation checks!
        """
        schema = SettingsSchema()
        for param in schema:
            #print(param)
            if "NUM_PARAMETERS" == param:
                continue
            assert param in cfg
            if schema[param][0] in (number, int, float):
                assert schema[param][1][0] <= cfg[param] <= schema[param][1][1], f"OOB: {param}={cfg[param]} is out-of-bounds. {schema[param][1][0]} <= {cfg[param]} <= {schema[param][1][1]}"
            else:
                assert cfg[param] in schema[param][1], f"Key: {cfg[param]} not found. Should be one of {schema[param][1]}"
        return cfg

class JSONParser(BaseParser):
    """Parses a configuration file in json format to Simulation format.

    The JSONParser takes an input file (path to file) in json format and attempts to parse. While parsing it validates the given parameters, check for completeness of parameters and performs various out-of-bounds checks.
    """

    def parse(self):
        with open(self.path, "r") as file:
            cfg = json.load(file)
        return cfg

class YAMLParser(BaseParser):
    """Parses a configuration file in YAML format to Simulation format.

    The YAMLParser takes an input file (path to file) in YAML format and attempts to parse. While parsing it validates the given parameters, check for completeness of parameters and performs various out-of-bounds checks.
    """

    def parse(self):
        with open(self.path, "r") as file:
            cfg = yaml.full_load(file)
        return cfg


if __name__ == "__main__":
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

    parser = JSONParser("./cfg/input.json")
    cfg = parser(validate=True)
    print("Parsed input is: ")
    print(cfg)
    if cfg == TEST_INPUT:
        print("JSONParser: [SUCCESS]")
    else:
        print("[JSONParser: FAILURE]")

    parser = YAMLParser("./cfg/input.yaml")
    cfg = parser(validate=True)
    print("Parsed input is: ")
    print(cfg)
    if cfg == TEST_INPUT:
        print("YAMLParser: [SUCCESS]")
    else:
        print("YAMLParser: [FAILURE]")
