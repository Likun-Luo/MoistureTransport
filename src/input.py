#!/usr/bin/env python3
import json
import pathlib
from dataclasses import dataclass


@dataclass
class MoiRaParser:
    """Parses a configuration file in json format to Simulation format.

    The MoiRaParser takes an input file (path to file) in json format and attempts to parse. While parsing it validates the given parameters, check for completeness of parameters and performs various out-of-bounds checks.
    """
    path: pathlib.Path

    def __call__(self):
        return self.validate(self.parse())

    def parse(self):
        with open(self.path, "r") as file:
            cfg = json.load(file)
        return cfg

    def validate(self, cfg):
        """validates a configuratione.

        Performs various checks on the cfg-object (python dictionary) based on the serialized json-file. Mainly, this involves checking whether the input is of valid format, size and also performs out of bounds checks.

        TODO: Implement validation checks!
        """
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

    parser = MoiRaParser("./cfg/input.json")
    cfg = parser()
    print("Parsed input is: ")
    print(cfg)
    if cfg == TEST_INPUT:
        print("[SUCCESS]")
    else:
        print("[FAILURE]")
