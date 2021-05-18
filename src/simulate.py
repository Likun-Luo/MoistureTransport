#!/usr/bin/env python3
"""simulation stuff

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

<module description.>
"""
# STL imports
import math
from time import sleep
# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
# interal imports
from .process import draw_placeholder

# These are example parameters
# All caps name to signal it's a constant (or should be treated as such),
# and so its not confused with the constructor argument.
SIM_PARAMS_EXAMPLE = {
    "material": "brick",
    "sampleLength": 0.2,
    "moistureUptakeCoefficient": 10.0,
    "freeSaturation": 300.0,
    "meanPoreSize": 1e-6,
    "freeParameter": 13,
    "numberofElements": 100,
    "timeStepSize": 0.01,
    "totalTime": 10,
    "Anfangsfeuchte": 40
}

# Classes should be named in upper camel case (MyExampleClass)...so this is fine
class Simulation:

    def __init__(self, sim_params):
        # You're using C-naming style.
        # Variables "should" all be lower case (snake_case_style) --> PEP8
        # I don't always conform to that either, e.g. matrices are often 
        # called A and then it can be nice to name the variable the same
        # ... technically, that isn't PEP8 though!
        # We "should" rename them like this
        #   A --> moisture_uptake_coefficient
        #   L --> sample_length / length
        # 
        # In the config file, these naming conventions are fine,
        # but maybe we should also do it like that there.
        self.moisture_uptake_coefficient = sim_params["moistureUptakeCoefficient"]
        self.length = sim_params["sampleLength"]
        # also, it's not good style to use abbreviations due to
        # usually being harder to understand for people other than you.
        #   free_sat --> free_saturation
        # Code completion is a standard feature in every editor for a reason ;)
        self.free_saturation = sim_params["freeSaturation"]
        self.pore_size = sim_params["meanPoreSize"]
        self.free_parameter = sim_params["freeParameter"]

    def w(self, P_suc):
        """water retention curve

        Parameters:
            P_suc ... suction pressure
        
        Returns:
            w ... liquid moisture content
        """

        return self.free_saturation / (1.0 + self.pore_size * P_suc)

    def P_suc(self, w):
        """Inverse of water retention curve

        Parameters:
            w ... current liquid moisture content

        Returns:
            P_suc ... suction pressure

        Raises:
            ValueError ... if w==0
        """

        if w != 0:
            return (self.free_saturation - w) / (self.pore_size * w)
        print(f"Error: {w}==0 --> division by zero!")
        # Just raise a built in Error here instead!
        raise ValueError(f"Error: {w} division by zero")

    def dw(self, P_suc):
        """Derivative of w(P_suc)

        Needed for the calculation of total moisture conductivity K_w

        Parameters:
            P_suc ... suction pressure

        Returns:
            dw ... derivative of w(P_suc) d w(P_suc)/d P_suc
        """
        return -self.free_saturation * self.pore_size / (self.pore_size * P_suc + 1.0)**2

    def K_w(self, P_suc):
        """total moisture conductivity Kw

        Parameters:
            P_suc ... suction pressure
        
        Returns:
            K_w ... total moisture conductivity Kw
        """

        const = (self.w(P_suc) / self.free_saturation)**self.free_parameter  #reuse data

        return -self.dw(P_suc) * ((self.free_parameter + 1) / (2 * self.free_parameter)) * (self.moisture_uptake_coefficient / self.free_saturation)**2 * \
            const * (self.free_parameter + 1 - const)

    # All method and function names should be lower case!
    # To be precise: snake_case_style
    # Same goes for all attributes actually, though I'd be more lenient here.
    # You know, matrices are often called A and then it can be nice to name
    # the variable the same... technically, that isn't PEP8 though.
    def draw(self):
        """compare the curve with literature

        stimmt nicht ganz
        """
        P_suc = np.linspace(0, 1e9, 100000)
        Kw = self.K_w(P_suc)
        # Ideally, we would do all plotting in process.py.
        # Might turn out to be not feasible or clunky though!
        draw_placeholder(P_suc, Kw) 

    def iterate(self):
        """run the simulation

        Just a placeholder for now...mainly to show tqdm.
        TQDM Usage, see: https://github.com/tqdm/tqdm#usage
        """
        print("Running first sweep")
        for i in trange(10):
            sleep(0.1)
        print("Running second sweep")
        for i in tqdm(range(10)):
            sleep(0.1)
        print("Simulation done! (Wow...that was fast!)")

    def printParams(self):
        print("Moisture uptake coefficient :", self.pore_size)

if __name__ == "__main__":
    x = Simulation(SIM_PARAMS_EXAMPLE)
    x.draw()
