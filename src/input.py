#!/usr/bin/env python3
"""Module for input parsing of a Simulation configuration file.

<one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2021  Holzner, Peter & Luo, Likun

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
# STL imports
import json
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Union
# 3rd party imports
import yaml
from numpy import inf
# interal imports

VALID_FILE_FORMATS = [".json", ".yaml", ".cfg"]
number = [int, float]


@dataclass
class SettingsSchema:
    # base material and geometry
    material: tuple = (str, ["brick", "cement", "wood"])  # 1
    sampleLength: tuple = (number, [1e-2, 1])  # in m, min=1cm  # 2
    moistureUptakeCoefficient: tuple = (number, [1e-1, 1000])  # 3
    freeSaturation: tuple = (number, [1, 1e6])  # 4
    meanPoreSize: tuple = (number, [1e-9, 1e-1])  # 5
    freeParameter: tuple = (number, [-inf, inf])  # 6
    # boundary & initial conditions
    Anfangsfeuchte: tuple = (number, [1e-2, 1000])  # 7
    # simulation parameters
    numberofElements: tuple = (int, [1, 1e3])  # 8
    timeStepSize: tuple = (number, [1e-6, 1])  # 9
    totalTime: tuple = (number, [1e-1, 1 * 24 * 3600])  # 10
    # seconds, 3600s = 1h
    averagingMethod: tuple = (str, ["linear", "harmonic"])  # 11
    # number of settings
    NUM_PARAMETERS: tuple = (int, [11, 11])

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

    def __call__(self, validate=True):
        if validate:
            return self.validateCFG(self.parse())
        #print("Input validation is turned off.")
        return self.parse()

    def parse(self):
        raise NotImplementedError("parse method not implemented!")

    @staticmethod
    def validateCFG(cfg):
        """validates a configuratione.

        Performs various checks on the cfg-object (python dictionary) based on the serialized json-file. Mainly, this involves checking whether the input is of valid format, size and also performs out of bounds checks.
        """
        schema = SettingsSchema()
        for param in schema:
            #print(param)
            if "NUM_PARAMETERS" == param:
                continue
            # Check if parameter is in schema? --> KeyError
            if param not in cfg:
                raise KeyError(f"parameter '{param}' not found in {cfg.keys()}")
            # Check if input parameter is of correct type? --> TypeError
            if schema[param][0] == number:
                if not any([isinstance(cfg[param], ptype) for ptype in schema[param][0]]):
                    raise TypeError(f"Value '{cfg[param]}' of parameter '{param}' should be a number (int or float)!")
            if schema[param][0] == int:
                if not isinstance(cfg[param], schema[param][0]):
                    raise TypeError(f"Value '{cfg[param]}' of parameter '{param}' should be an integer number (int)!")
            if schema[param][0] == float:
                if not isinstance(cfg[param], schema[param][0]):
                    raise TypeError(f"Value '{cfg[param]}' of parameter '{param}' should be a floating point number (float)!")
            if schema[param][0] == str:
                if not isinstance(cfg[param], schema[param][0]):
                    raise TypeError(f"Value '{cfg[param]}' of parameter '{param}' should be a string (str)!")
            # Check if input parameter value is within bounds/choices? --> ValueError
            if schema[param][0] in (number, int, float):
                if not (schema[param][1][0] <= cfg[param] <= schema[param][1][1]):
                    raise ValueError(f"OOB: {param}={cfg[param]} is out-of-bounds. {schema[param][1][0]} <= {cfg[param]} <= {schema[param][1][1]}")
            else:
                if cfg[param] not in schema[param][1]:
                    raise ValueError(f"Key: {cfg[param]} not found. Should be one of {schema[param][1]}")
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

DefaultParser = JSONParser
