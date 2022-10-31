"""
CEASIOMpy: Conceptual Aircraft Design Software

Description : Create a set of airfoil coordinates using CST parametrization method

Input  : wl = CST weight of lower surface
         wu = CST weight of upper surface
         dz = trailing edge thickness

Output : coord = set of x-y coordinates of airfoil generated by CST

Adapted from: Kulfan_CST/kulfan_to_coord.py
              by Ryan Barrett 'ryanbarr'
              https://github.com/Ry10/Kulfan_CST

 Adapted from: Airfoil generation using CST parameterization method
               by Pramudita Satria Palar
               http://www.mathworks.com/matlabcentral/fileexchange/42239-airfoil-generation-using-cst-parameterization-method

Python version: >=3.8

| Author: Aidan Jungo
| Creation: 2021-04-26

TODO:

    *

"""

# =================================================================================================
#   IMPORTS
# =================================================================================================

from math import cos, factorial, pi
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).parent

# =================================================================================================
#   CLASSES
# =================================================================================================


class CST_shape(object):
    def __init__(self, wl=None, wu=None, dz=0, N=200):

        if wl is None:
            self.wl = [-1, -1, -1]
        else:
            self.wl = wl

        if wu is None:
            self.wu = [1, 1, 1]
        else:
            self.wu = wu

        self.dz = dz
        self.N = N
        self.x_list = []
        self.y_list = []
        self.coordinate = np.zeros(N)

    def airfoil_coor(self):
        wl = self.wl
        wu = self.wu
        dz = self.dz
        N = self.N

        # Create x coordinate
        x = np.ones((N, 1))
        y = np.zeros((N, 1))
        zeta = np.zeros((N, 1))

        for i in range(0, N):
            zeta[i] = 2 * pi / N * i
            x[i] = 0.5 * (cos(zeta[i]) + 1)

        # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
        N1 = 0.5
        N2 = 1

        # Used to separate upper and lower surfaces
        center_loc = np.where(x == 0)
        center_loc = center_loc[0][0]

        xl = np.zeros(center_loc)
        xu = np.zeros(N - center_loc)

        # Lower surface x-coordinates
        for i in range(len(xl)):
            xl[i] = x[i]

        # Upper surface x-coordinates
        for i in range(len(xu)):
            xu[i] = x[i + center_loc]

        # Call ClassShape function to determine lower and upper surface y-coordinates
        yl = self.__ClassShape(wl, xl, N1, N2, -dz)
        yu = self.__ClassShape(wu, xu, N1, N2, dz)

        # Combine upper and lower y coordinates
        y = np.concatenate([yl, yu])

        self.coord = [x, y]

        self.x_list = x.ravel().tolist()
        self.y_list = y.ravel().tolist()

        # self.plotting()
        return self.coord

    @staticmethod
    def __ClassShape(w, x, N1, N2, dz):

        # Class function; taking input of N1 and N2
        C = np.zeros(len(x))
        for i in range(len(x)):
            C[i] = x[i] ** N1 * ((1 - x[i]) ** N2)

        # Shape function; using Bernstein Polynomials
        n = len(w) - 1  # Order of Bernstein polynomials

        K = np.zeros(n + 1)
        for i in range(0, n + 1):
            K[i] = factorial(n) / (factorial(i) * (factorial((n) - (i))))

        S = np.zeros(len(x))
        for i in range(len(x)):
            S[i] = 0
            for j in range(0, n + 1):
                S[i] += w[j] * K[j] * x[i] ** (j) * ((1 - x[i]) ** (n - (j)))

        # Calculate y output
        y = np.zeros(len(x))
        for i in range(len(y)):
            y[i] = C[i] * S[i] + x[i] * dz

        return y


# =================================================================================================
#    MAIN
# =================================================================================================

if __name__ == "__main__":

    print("Nothing to execute!")
