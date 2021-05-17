import numpy as np
import matplotlib.pyplot as plt
import math

sim_params = {
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


class Simulation:

    def __init__(self, sim_params):
        self.A = sim_params["moistureUptakeCoefficient"]
        self.L = sim_params["sampleLength"]
        self.freeSat = sim_params["freeSaturation"]
        self.PoreSize = sim_params["meanPoreSize"]
        self.n = sim_params["freeParameter"]

    def w(self, P_suc):
        """water retention curve

        Parameters:
            P_suc ... suction pressure
        
        Returns:
            w ... liquid moisture content
        """

        return self.freeSat / (1.0 + self.PoreSize * P_suc)

    def P_suc(self, w):
        """Inverse of water retention curve

        Parameters:
            w ... current liquid moisture content

        Returns:
            P_suc ... suction pressure
        """

        if w != 0:
            return (self.freeSat - w) / (self.PoreSize * w)
        else:
            print("Error, division by zero")

    def dw(self, P_suc):
        """Derivative of w(P_suc)

        Needed for the calculation of total moisture conductivity K_w

        Parameters:
            P_suc ... suction pressure

        Returns:
            dw ... derivative of w(P_suc) d w(P_suc)/d P_suc
        """
        return -self.freeSat * self.PoreSize / (self.PoreSize * P_suc + 1.0)**2

    def K_w(self, P_suc):
        """total moisture conductivity Kw

        Parameters:
            P_suc ... suction pressure
        
        Returns:
            K_w ... total moisture conductivity Kw
        """

        const = (self.w(P_suc) / self.freeSat)**self.n  #reuse data

        return -self.dw(P_suc) * ((self.n + 1) / (2 * self.n)) * (self.A / self.freeSat)**2 * \
            const * (self.n + 1 - const)

    def Draw(self):
        """compare the curve with literature

        stimmt nicht ganz
        """
        P_suc = np.linspace(0, 1e9, 100000)
        w = self.K_w(P_suc)
        plt.plot(P_suc, w)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    def printParams(self):
        print("Moisture uptake coefficient :", self.PoreSize)


x = Simulation(sim_params)
x.Draw()
