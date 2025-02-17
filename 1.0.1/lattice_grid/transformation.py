#Electromagnetic Particle-In-Cell Method
#Developer and Researcher: Christopher Emmanuelle J. Visperas

import numpy as np

class Plane:
    def __init__(self, x):
        #coordinates
        self.x = x[0]
        self.y = x[1]

    def cartesian(self):
        return self.x, self.y
        
    def polar(self):
        r = np.sqrt(np.power(self.x, 2) + np.power(self.y, 2) + np.power(self.z, 2))
        theta = np.arctan2(self.y, self.x)
        return r, theta
        
class Solid:
    def __init__(self, x):
            
        #initial x, y, z coordinate
        self.x = x[0]
        self.y = x[1]
        self.z = x[2]

    def cartesian(self):
        return self.x, self.y, self.z
        
    def spherical(self):
        r = np.sqrt(np.power(self.x, 2) + np.power(self.y, 2) + np.power(self.z, 2))
        theta = np.arctan2(self.y, self.x)
        phi = np.arccos(self.z/r)
        return r, theta, phi
        
    def cylindrical(self):
        rho = np.sqrt(np.power(self.x, 2) + np.power(self.y, 2))
        theta = np.arctan2(self.y, self.x)
        return rho, theta, self.z
