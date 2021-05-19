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
# 3rd party imports

# interal imports
from src.input import DefaultParser
from src.simulate import Simulation
#from src.process import *

SIM_VERSION = "0.1"

argParser = ArgumentParser(prog=f"MoistureTransport Simulation v{SIM_VERSION}",
                           description="blabla",
                           prefix_chars="?")

cfgParser = DefaultParser("./cfg/input.json")
cfg = cfgParser()

sim = Simulation(cfg)
print("Drawing starting state")
sim.draw()
sim.demo()
print("Drawing final state")
sim.draw()
sim.draw_watercontent()