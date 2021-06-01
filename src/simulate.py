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
from os import error
from time import sleep
# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
# interal imports
## TODO: Remove before release
if __name__ == "__main__":
    from process import draw_placeholder, draw_watercontent
else:
    from .process import draw_placeholder, draw_watercontent

SIM_PARAMS_EXAMPLE = {
    "material": "brick",
    "sampleLength": 0.2,
    "moistureUptakeCoefficient": 10.0,
    "freeSaturation": 300.0,
    "meanPoreSize": 1e-6,
    "freeParameter": 13,
    "numberofElements": 20,
    #"timeStepSize": 0.01,
    "totalTime": 10,
    "Anfangsfeuchte": 1,
    "averagingMethod": "linear"
}


class Simulation:

    def __init__(self, sim_params):
        self.moisture_uptake_coefficient = sim_params[
            "moistureUptakeCoefficient"]
        self.length = sim_params["sampleLength"]
        self.free_saturation = sim_params["freeSaturation"]
        self.pore_size = sim_params["meanPoreSize"]
        self.free_parameter = sim_params["freeParameter"]
        self.number_of_element = sim_params["numberofElements"]
        self.initial_moisture_content = sim_params["Anfangsfeuchte"]
        self.averaging_method = sim_params["averagingMethod"]

        self.dx = self.length / self.number_of_element
        #self.dt = sim_params["timeStepSize"]
        self.total_time = sim_params["totalTime"]
        # Don't do it this way: wastes a lot of memory for large dt and totalTime
        #self.time_range = np.arange(0, sim_params["totalTime"] + self.dt,
        #                            self.dt)

        self.current_time = .0
        self.dt_init = 1e-10

        self.w_control_volume = np.zeros(self.number_of_element + 2)

    def initial_condition(self):
        """parse initial condition to the control volumes
        """

        self.w_control_volume[:] = self.initial_moisture_content / 100 * self.free_saturation

        self.w_control_volume[0] = self.free_saturation  # Left hand side
        self.w_control_volume[self.number_of_element +
                              1] = self.w_control_volume[
                                  self.number_of_element]  # Right hand side
        self.current_dt = self.dt_init

    def w(self, P_suc):
        """water retention curve

        Parameters:
            P_suc ... suction pressure
        
        Returns:
            w ... liquid moisture content
        """
        val = (1.0 + self.pore_size * P_suc)

        ret = self.free_saturation / val

        #if val < 1e-5:
        #    print("val: ", val)
        #    print("pore_size: ", self.pore_size)
        #    print("P_suc: ", P_suc)

        return ret

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
        raise ValueError(f"Error: {w} division by zero")

    def dw(self, P_suc):
        """Derivative of w(P_suc)

        Needed for the calculation of total moisture conductivity K_w

        Parameters:
            P_suc ... suction pressure

        Returns:
            dw ... derivative of w(P_suc) d w(P_suc)/d P_suc
        """
        return -self.free_saturation * self.pore_size / (
            self.pore_size * P_suc + 1.0)**2

    def K_w(self, P_suc):
        """total moisture conductivity Kw

        Parameters:
            P_suc ... suction pressure
        
        Returns:
            K_w ... total moisture conductivity Kw

        TODO: Wrap with np.vectorize? (https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html)
        """

        const = (self.w(P_suc) /
                 self.free_saturation)**self.free_parameter  #reuse data

        # l1 = ((self.free_parameter + 1) / (2 * self.free_parameter))
        # l2 = (self.moisture_uptake_coefficient / self.free_saturation)**2
        # l3 = const * (self.free_parameter + 1 - const)

        l1 = const * (self.free_parameter + 1) / (2 * self.free_parameter) * (
            self.free_parameter + 1 - const)
        l2 = (self.moisture_uptake_coefficient / self.free_saturation)**2

        return -self.dw(P_suc) * l1 * l2

    def K_interface(self, K_P, K_W):
        """calculate the liquid conductivity at the interface between two nodes

        Parameters:
            K_P ... nodal moisture conductivity at one node P

            K_W ... nodal moisture conductivity at the neighbour node W

        Returns:
            K_interface ... liquid conductivity at the interface

        Raises:
            ValueError ... averaging_method not one of [linear, harmonic]
        """

        if self.averaging_method == "linear":
            return (K_P + K_W) / 2
        if self.averaging_method == "harmonic":
            return 2 * K_W * K_P / (K_W + K_P)
        raise ValueError(
            f"averaging_method={self.averaging_method} not one of [linear, harmonic]!"
        )

    def dwdt(self, w_P_C, w_P_W, w_P_E):
        """evaluate the right hand side of the governing equation (time derivative of w_P)
        """

        P_suc_P = self.P_suc(w_P_C)
        P_suc_W = self.P_suc(w_P_W)
        P_suc_E = self.P_suc(w_P_E)

        K_e = self.K_interface(self.K_w(P_suc_P), self.K_w(P_suc_E))
        K_w = self.K_interface(self.K_w(P_suc_P), self.K_w(P_suc_W))

        dwdt = -K_e * (P_suc_E - P_suc_P) / self.dx**2 - K_w * (
            P_suc_W - P_suc_P) / self.dx**2

        return dwdt

    def dwdt_old(self, w_P, index):
        """evaluate the right hand side of the governing equation (time derivative of w_P)
        """

        P_suc_P = self.P_suc(w_P)
        P_suc_W = self.P_suc(self.w_control_volume[index - 1])
        P_suc_E = self.P_suc(self.w_control_volume[index + 1])

        K_e = self.K_interface(self.K_w(P_suc_P), self.K_w(P_suc_E))
        K_w = self.K_interface(self.K_w(P_suc_P), self.K_w(P_suc_W))

        dwdt = -K_e * (P_suc_E - P_suc_P) / self.dx**2 - K_w * (
            P_suc_W - P_suc_P) / self.dx**2

        return dwdt

    def rk5(self):
        """runge kutta 5 method with local error estimation
        """

        volume = self.w_control_volume

        with np.nditer([volume[:-2], volume[1:-1], volume[2:]],
                       op_flags=['readwrite'],
                       flags=["f_index"],
                       order="C") as it:
            for w_P_W, w_P, w_P_E in it:

                k1 = self.dwdt(w_P, w_P_W, w_P_E)
                k2 = self.dwdt((w_P + 0.25 * self.current_dt * k1), w_P_W,
                               w_P_E)
                k3 = self.dwdt((w_P + 4 * self.current_dt * k1 / 81 +
                                32 * self.current_dt * k2 / 81), w_P_W, w_P_E)
                k4 = self.dwdt((w_P + 57 * self.current_dt * k1 / 98 -
                                432 * self.current_dt * k2 / 343 +
                                1053 * self.current_dt * k3 / 686), w_P_W,
                               w_P_E)
                k5 = self.dwdt((w_P + 1 * self.current_dt * k1 / 6 +
                                27 * self.current_dt * k3 / 52 +
                                49 * self.current_dt * k4 / 156), w_P_W, w_P_E)

                rhs = self.current_dt * (43 * k1 / 288 + 243 * k3 / 416 +
                                         343 * k4 / 1872 + k5 / 12)

                error = self.current_dt * (-5 * k1 / 288 + 27 * k3 / 416 -
                                           245 * k4 / 1872 + k5 / 12)

                if w_P + rhs > self.free_saturation or error > 1e-6:
                    pass

    def rk5_new(self, w_P_W, w_P, w_P_E):
        """runge kutta 5 method with local error estimation
        """

        k1 = self.dwdt(w_P, w_P_W, w_P_E)
        k2 = self.dwdt((w_P + 0.25 * self.current_dt * k1), w_P_W, w_P_E)
        k3 = self.dwdt((w_P + 4 * self.current_dt * k1 / 81 +
                        32 * self.current_dt * k2 / 81), w_P_W, w_P_E)
        k4 = self.dwdt((w_P + 57 * self.current_dt * k1 / 98 -
                        432 * self.current_dt * k2 / 343 +
                        1053 * self.current_dt * k3 / 686), w_P_W, w_P_E)
        k5 = self.dwdt(
            (w_P + 1 * self.current_dt * k1 / 6 +
             27 * self.current_dt * k3 / 52 + 49 * self.current_dt * k4 / 156),
            w_P_W, w_P_E)

        rhs = self.current_dt * (43 * k1 / 288 + 243 * k3 / 416 +
                                 343 * k4 / 1872 + k5 / 12)

        error = self.current_dt * (-5 * k1 / 288 + 27 * k3 / 416 -
                                   245 * k4 / 1872 + k5 / 12)

        return rhs, error

    def simulation_test(self):

        self.initial_condition()

        self.t = []
        self.w_total = []

        cnt = 0

        print("initial w = %.3f" % self.total_moisture_sample(flag="relative"), "%")
        print("number of element: ",self.number_of_element)
        print("initial timestep: ", self.current_dt)

        while self.current_time < self.total_time:

            valid = False

            volume = self.w_control_volume

            with np.nditer([volume[:-2], volume[1:-1], volume[2:]],
                           op_flags=['readwrite'],
                           flags=["f_index"],
                           order="C") as it:
                for w_P_W, w_P, w_P_E in it:
                    rhs, error = self.rk5_new(w_P_W, w_P, w_P_E)
                    if (w_P + rhs > self.free_saturation) or (error > 1e-6):
                        valid = False
                        self.current_dt /= 2
                        break

                    elif (rhs > 1e-12):
                        w_P += rhs
                        valid = True

                    else:
                        valid = True
                        break

            if valid:

                if cnt % 100 == 0:
                    self.t.append(self.current_time)
                    self.w_total.append(
                        self.total_moisture_sample(flag="relative"))

                cnt += 1
                self.current_time += self.current_dt
                self.current_dt *= 2

            print("Progress: %.2f / %d, current time step: %.2e s" %
                  (self.current_time, self.total_time, self.current_dt),
                  end="\r")

        print("final time step: ", self.current_dt)
        print("final w = %.3f" % self.total_moisture_sample(flag="relative"), "%")

        title = "Number of elements = " + str(
            self.number_of_element) + ",initial time step = " + str(
                self.dt_init) + ",final time step = " + str(
                    self.current_dt) + ", initial saturation = " + str(
                        self.initial_moisture_content) + "%"
        # f-string version (often shorter or at least as short):
        #title = f"Number of elements = {self.number_of_element}, time step = {self.dt}, initial saturation = {self.initial_moisture_content}%"

        # Make this a method (or function in process.py)
        # --> Smaller blocks of code == easier to understand and maintain!
        plt.plot(self.t, self.w_total, label="linear averaging")
        plt.title(title)
        plt.ylabel("saturation degree [%]")
        plt.xlabel("time [hours]")
        plt.legend()
        plt.show()

    def runge_kutta(self):
        """runge kutta method to calculate the moisture content at a specific control volume
        """

        volume = self.w_control_volume
        #for index in range(1, self.number_of_element + 1):
        # for index, (w_P_W, w_P, w_P_E) in enumerate(zip(volume[:1], volume[1:], volume[2:])):
        # Iterating over numpy arrays: https://numpy.org/doc/stable/reference/arrays.nditer.html#tracking-an-index-or-multi-index
        with np.nditer([volume[:-2], volume[1:-1], volume[2:]],
                       op_flags=['readwrite'],
                       flags=["f_index"],
                       order="C") as it:
            for w_P_W, w_P, w_P_E in it:

                # w_P = self.w_control_volume[index]

                # k1 = self.dwdt(w_P, index)
                k1 = self.dwdt(w_P, w_P_W, w_P_E)
                k2 = self.dwdt((w_P + 0.5 * self.dt * k1), w_P_W, w_P_E)
                k3 = self.dwdt((w_P + 0.5 * self.dt * k2), w_P_W, w_P_E)
                k4 = self.dwdt((w_P + 1.0 * self.dt * k3), w_P_W, w_P_E)

                rhs = self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

                # if (rhs >= 1e-12):
                #     self.w_control_volume[index] += rhs
                # else:
                #     break
                # Same logic, but a bit faster AND imo easier to understand
                if (rhs < 1e-12):
                    break
                w_P += rhs
                # self.w_control_volume[index] += rhs

    # Probe [ger] == sample [engl]
    def total_moisture_sample(self, flag="absolute"):
        """calculate the total moisture of the sample
        """
        w = 0

        # Index access is terribly slow in pyton
        # for i in range(1,self.number_of_element + 1):
        #     w += self.w_control_volume[i] * self.dx

        # Iterate over the elements directly (faster):
        # for item in self.w_control_volume[1:]:
        #     w += item * self.dx

        # OR use the even faster list comprehension feature (even faster):
        # Allows the interpreter to do some optimization...
        # w = sum([self.dx*item for item in self.w_control_volume[1:]])

        # OR use numpy (FASTEST): --> C-Code
        # Reasoning being that Python is slow and C is fast.
        # (numpy is just compiled C-Code)
        w = np.sum(self.dx * self.w_control_volume[1:-1])

        if flag == "absolute":
            return w / self.length
        else:
            return 100 * w / (self.free_saturation * self.length)

    def run_simulation(self):

        self.initial_condition()

        self.t = []
        self.w_total = []

        cnt = 0

        print("w = %.3f" % self.total_moisture_sample(flag="relative"))

        # Just use the generator expression here, so it doesn't use
        # as much memory and is gone after the loop.
        # In Python, how you loop makes a huge difference!
        # Index access is fast in C/C++, but is the worst in Python!
        #for t in self.time_range:
        for t in tqdm(np.arange(0, self.total_time + self.dt, self.dt)):

            self.runge_kutta()

            if cnt % 100 == 0:
                self.t.append(t)
                self.w_total.append(self.total_moisture_sample(flag="relative"))

            cnt += 1
            # I prefer the newer f-strings in python:
            # print(f"progress: {t:.2f} / {self.total_time}", end="\r")
            # but use whichever you prefer!
            #print("progress: %.2f / %d" % (t, self.total_time), end="\r")

        #print(self.w_control_volume)

        print("w = %.2f" % self.total_moisture_sample(flag="relative"))

        title = "Number of elements = " + str(
            self.number_of_element) + ", time step = " + str(
                self.dt) + ", initial saturation = " + str(
                    self.initial_moisture_content) + "%"
        # f-string version (often shorter or at least as short):
        #title = f"Number of elements = {self.number_of_element}, time step = {self.dt}, initial saturation = {self.initial_moisture_content}%"

        # Make this a method (or function in process.py)
        # --> Smaller blocks of code == easier to understand and maintain!
        plt.plot(self.t, self.w_total, label="linear averaging")
        plt.title(title)
        plt.ylabel("saturation degree [%]")
        plt.xlabel("time [hours]")
        plt.legend()
        plt.show()

    def iterate(self):
        self.run()

    def draw(self):
        """compare the curve with literature

        stimmt nicht ganz
        """
        P_suc = np.linspace(0, 1e9, 100000)
        Kw = np.vectorize(self.K_w)(P_suc)
        draw_placeholder(P_suc, Kw)

    def draw_watercontent(self):
        """watercontent

        demo
        """
        t = np.arange(0, self.total_time, self.dt)

        w = 10 * np.sqrt(t)
        w_last = w[-1] * np.ones_like(w)
        w = np.append(w, w_last)
        t = np.arange(0, 2 * self.total_time, self.dt)
        draw_watercontent(w, t)

    def demo(self):
        """demo the simulation

        Just a placeholder for now...mainly to show tqdm.
        TQDM Usage, see: https://github.com/tqdm/tqdm#usage
        """
        print("Running first sweep")
        for i in trange(10):
            sleep(0.1)
        # print("Running second sweep")
        # for i in tqdm(range(10)):
        #     sleep(0.1)
        print("Simulation done! (Wow...that was fast!)")

    def print_params(self):
        print("Moisture uptake coefficient :", self.pore_size)


if __name__ == "__main__":
    x = Simulation(SIM_PARAMS_EXAMPLE)
    #x.run_simulation()
    x.simulation_test()
