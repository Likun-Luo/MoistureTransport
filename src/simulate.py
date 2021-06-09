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

EPS = np.finfo(float).eps

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
        self.total_time = sim_params["totalTime"]

        self.current_time = .0
        self.dt_init = 1e-10
        self.current_dt = self.dt_init

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

        raise ValueError(
            f"averaging_method={self.averaging_method} not yet implemented!"
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

    def rk5(self, w_P_W, w_P, w_P_E):
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

    def total_moisture_sample(self, flag="absolute"):
        """calculate the total moisture of the sample
        """
        w = np.sum(self.dx * self.w_control_volume[1:-1])

        if flag == "absolute":
            return w / self.length
        else:
            return 100 * w / (self.free_saturation * self.length)

    def update(self):
        """update the control volume.

        calculates one iteration step and updates the control volume.
        Validity of the update value is checked.
        """

        volume = self.w_control_volume  # for better format/readability
        valid = True  # It's valid unless the one break condition is true
        
        with np.nditer([volume[:-2], volume[1:-1], volume[2:]],
                       op_flags=['readwrite'],
                       #op_flags=['read'],
                       flags=["f_index"],
                       order="C") as it:
            for w_P_W, w_P, w_P_E in it:
                rhs, error = self.rk5(w_P_W, w_P, w_P_E)
                # A little bit more concise ;)
                if (w_P + rhs > self.free_saturation) or (error > 1e-6):
                    valid = False
                    self.current_dt /= 2
                    break
                if (rhs < 1e-12):
                    break
                w_P += rhs
        return valid

    def update_buffered(self, buffer):
        """update the control volume.

        calculates one iteration step and updates the control volume.
        Validity of the update value is checked.
        """

        volume = self.w_control_volume  # for better format/readability
        valid = True  # It's valid unless the one break condition is true
        
        with np.nditer([volume[:-2], volume[1:-1], volume[2:], buffer],
                       op_flags=['readwrite'],
                       #op_flags=['read'],
                       flags=["f_index"],
                       order="C") as it:
            for w_P_W, w_P, w_P_E, buf in it:
                rhs, error = self.rk5(w_P_W, w_P, w_P_E)
                # A little bit more concise ;)
                if (w_P + rhs > self.free_saturation) or (error > 1e-6):
                    valid = False
                    self.current_dt /= 2
                    break
                if (rhs < 1e-12):
                    break
                buf = rhs
                #w_P += rhs
        if valid:
            volume[1:-1] = volume[1:-1] + buffer
        return valid

    def simulation_test(self):
        """Run a simulation with adaptive iteration scheme.
        """
        self.initial_condition()
        print("initial w = %.3f" % self.total_moisture_sample(flag="relative"),
              "%")
        print("number of element: ", self.number_of_element)
        print("initial timestep: ", self.current_dt)
        print("total simulation time: ", self.total_time)

        self.t = []
        self.w_total = []
        self.w_absolute = []
        cnt = 0
        #buffer = np.zeros_like(self.w_control_volume[1:-1])

        timestep_text = lambda: f"current time step = {self.current_dt:.2e}s"
        with tqdm(
                desc="Time: ",
                total=self.total_time,
                unit="it",
                ncols=150,
                mininterval=1,
                unit_scale=1,
                postfix=timestep_text(),
                bar_format="{desc}: {n:.2f}h --> {percentage:3.0f}%| "
                "{bar}| "
                "{n_fmt}/{total_fmt} [Projected runtime: {elapsed}<{remaining}, ' '{rate_fmt}{postfix}]",
                position=0) as progress:
            #[default: '{l_bar}{bar}{r_bar}'], where l_bar='{desc}: {percentage:3.0f}%|' and r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]' Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage, elapsed, elapsed_s, ncols, nrows, desc, unit, rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv, rate_inv_fmt, postfix, unit_divisor, remaining, remaining_s, eta. Note that a trailing ": " is automatically removed after {desc} if the latter is empty.
            while self.current_time < self.total_time:
                valid = self.update()
                #valid = self.update_buffered(buffer)

                if valid:
                    if cnt % 100 == 0:
                        self.t.append(self.current_time)
                        self.w_total.append(
                            self.total_moisture_sample(flag="relative"))
                        self.w_absolute.append(self.total_moisture_sample(flag="absolute"))
                        #plt.show()

                    cnt += 1
                    self.current_time += self.current_dt
                    self.current_dt *= 2

                # Progress output
                #percentage = self.current_time / self.total_time * 100
                # print(
                #     f"Progress: {self.current_time:.2f} / {self.total_time:d} ({percentage:3.0f}%), current time step: {self.current_dt:.2e} s",
                #     end="\r")
                progress.postfix = timestep_text()
                #progress.percentage = percentage if percentage <= 100 else 99
                progress.n = self.current_time if self.current_time < self.total_time else self.total_time - 1  # * (1 - 5*EPS)
                progress.update()
        
        #print()
        results = self.analyze()
        self.show_results(results)

        self.plot_moisture()
        self.plot_moisture(sqrt_time=True)

    def plot_moisture(self, sqrt_time=False):
        """plots the current moisture content in the volume.
        """
        fig = plt.figure(figsize=(8,6))
        ax = plt.gca()
        
        if sqrt_time:
            plt.plot(np.sqrt(self.t), self.w_absolute, label=self.averaging_method)
            plt.xlabel("sqrt(time) [sqrt(hours)]")
            plt.ylabel("total moisture content [kg/m³]")
            fig.suptitle("Total moisture content over square root of time", fontsize=16, fontweight='bold') # fontstyle='italic'
        else:
            plt.plot(self.t, self.w_total, label=self.averaging_method)
            plt.xlabel("time [hours]")
            plt.ylabel("saturation [%]")
            fig.suptitle("Total saturation over time", fontsize=16, fontweight='bold')

        ax_title = f"N = {self.number_of_element}, dt = [{self.dt_init}, {self.current_dt}], initial saturation = {self.initial_moisture_content}%"
        ax.set_title(ax_title)
        
        plt.legend()
        plt.grid(True)
        plt.show()

    def analyze(self):

        results = {"A": 1}
        
        k = 0
        lower_limit = 0.2 * self.free_saturation * self.length
        upper_limit = 0.8 * self.free_saturation * self.length
        idx_lower = 0 
        idx_upper = 0 

        for i, w in enumerate(self.w_absolute):
            if w >= lower_limit and not idx_lower:
                idx_lower = i
            if w >= upper_limit and idx_lower:
                idx_upper = i
                break
        else:
            idx_upper = len(self.w_total) - 1 

        k = (self.w_absolute[idx_upper] - self.w_absolute[idx_lower]) * self.length / (np.sqrt(self.t[idx_upper]) - np.sqrt(self.t[idx_lower]))

        results["A"] = k
        return results

    def show_results(self, results):
        """prints final results.
        """
        print("------- SIMULATION DONE  -------")
        moist = self.total_moisture_sample(flag="relative")
        print("Final time step: ", self.current_dt)
        print(f"Final Total Moisture Saturation = {moist:.3f}", "%")
        print(f"Moisture Uptake Coefficient 'A' = {results['A']:.2f} kg/(m^2 h^0.5)")


if __name__ == "__main__":
    x = Simulation(SIM_PARAMS_EXAMPLE)
    x.simulation_test()
    # tot = 100
    # t = np.linspace(0, tot/2, int(tot/2/0.01))

    # w = 30 + 10 * np.sqrt(t)
    # w_last = w[-1] * np.ones_like(w)
    # w = np.append(w, w_last)
    # t = np.linspace(0, tot, int(tot/0.01))

    # x.t = t
    # x.w_total =w
    # x.plot_moisture()
    # x.plot_moisture(True)

    # x.show_results(x.analyze())
