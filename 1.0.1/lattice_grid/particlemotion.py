import numpy as np

class Motion:
    def __init__(self, xi : float, xf : float):
        #initial x, y, z coordinate
        self.xa = xi[0]
        self.ya = xi[1]
        self.za = xi[2]

        #final x, y, z coordinate
        self.xb = xf[0]
        self.yb = xf[1]
        self.zb = xf[2]
    
    def cubic_spline_gauss(self, x):
        