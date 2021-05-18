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


def draw_placeholder(P_suc, w, title="Kw(P_suc)"):
    """compare the curve with literature

    stimmt nicht ganz
    """
    plt.plot(P_suc, w)

    # Plot design
    plt.grid(True)

    plt.xscale('log')
    plt.yscale('log')

    plt.title(title)

    plt.xlabel("P_suc")
    plt.ylabel("Kw")

    plt.show()

def draw_watercontent(w, t):
    """compare the curve with literature

    stimmt nicht ganz
    """

    fig, ax = plt.subplots(1,2, figsize=(8,6))

    ax[0].plot(t, w)
    ax[0].set_xlabel("t [s]")

    ax[1].plot(np.sqrt(t), w)
    ax[1].set_xlabel("sqrt(t) [sqrt(h)]")

    # Plot design
    for a in ax:
        a.set_ylabel("w [%]")
        a.grid(True)
    plt.suptitle("Water content coefficient over time")
    plt.show()