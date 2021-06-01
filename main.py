#!/usr/bin/env python3
"""MoistureTransport Simulation 

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

The main program file
"""
# STL imports
from argparse import ArgumentParser
from pathlib import Path
# 3rd party imports

# interal imports
from src import *  # basic app configs - not simulation parameters!
from src.input import DefaultParser, YAMLParser, VALID_FILE_FORMATS
from src.simulate import Simulation
#from src.process import *

SIM_VERSION = "0.1"
welcome_text = f"""
#################################
# Moisture Transport Simulation #
#################################

Version: {SIM_VERSION}
Authors: - HOLZNER, Peter
         - LUO, Likun
"""
argParser = ArgumentParser(prog=f"MoistureTransport Simulation v{SIM_VERSION}",
                           description=welcome_text,
                           prefix_chars="-")

argParser.add_argument('--cfg', nargs="?", default="./cfg/input.json")
argParser.add_argument('--mode',
                       nargs="?",
                       choices=["demo", "uptake"],
                       default="uptake")
print(welcome_text)
args = argParser.parse_args()

######################
# Configuration file #
######################

# Prompt for cfg file
cfg_file_path = Path(args.cfg)
if not cfg_file_path.is_file():
    print(f"Provided file '{cfg_file_path}' isn't a file or can't be found!")
if cfg_file_path.suffix not in VALID_FILE_FORMATS:
    print(
        f"Provided file '{cfg_file_path}' does not have a valid format: {VALID_FILE_FORMATS}!"
    )

use_standard_cfg = None
while True:
    use_standard_cfg = input(
        f"Use the configuration file found at {cfg_file_path}? (y/N): ")

    if use_standard_cfg == "":
        print("User stopped: Simulation aborted...")
        exit(1)

    if use_standard_cfg in ["y", "Y"]:
        break

    if use_standard_cfg in ["n", "N"]:
        alt_file = Path(input("Path to configuration file: "))
        if alt_file.is_file():
            if alt_file.suffix in VALID_FILE_FORMATS:
                cfg_file_path = alt_file
                break
            print(
                f"Provided file '{alt_file}' does not have a valid format: {VALID_FILE_FORMATS}!"
            )
        else:
            print(f"Provided file '{alt_file}' isn't a file or can't be found!")

    print("Invalid input. Please use either y or N!")
# Cfg selected
print()
print("Parsing selected simulation configuration file...")
if cfg_file_path.suffix == ".json":
    cfgParser = DefaultParser(cfg_file_path)
elif cfg_file_path.suffix == ".yaml":
    cfgParser = YAMLParser(cfg_file_path)
else:
    raise ValueError(f"{cfg_file_path.suffix} is not a valid file format!")
cfg = cfgParser()
print("--> Parameters are valid!")
print()
#exit(0)
######################
# --- Simulation --- #
######################
mode = args.mode
print("mode: ", mode)
print("------- STARTING SIMULATION -------")
sim = Simulation(cfg)
print(f"Simulating a time span of:{sim.total_time} ")

if mode == "demo":
    print("Drawing starting state")
    sim.draw()

    sim.demo()

    print("Drawing final state")
    #sim.draw()
    sim.draw_watercontent()
else:
    sim.simulation_test()

print("------- SIMULATION DONE  -------")
print("Results:")
print("\tA = 1")
print("\tB = 1")
print("\t...")
print(f"Graphs and simulation report saved at: {RESULTS_DIR}")
