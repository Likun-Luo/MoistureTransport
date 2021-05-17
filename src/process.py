"""Module for post-processing and plotting of a MoistureTransport Simulation.

"""
#!/usr/bin/env python3
"""<module one liner> 

<module description.>

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

# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt
# interal imports


def draw_placeholder(P_suc, w):
    """compare the curve with literature

    stimmt nicht ganz
    """
    plt.plot(P_suc, w)

    # Plot design
    plt.grid(True)

    plt.xscale('log')
    plt.yscale('log')

    plt.title("P_suc(w)")

    plt.xlabel("P_suc")
    plt.ylabel("w")

    plt.show()
