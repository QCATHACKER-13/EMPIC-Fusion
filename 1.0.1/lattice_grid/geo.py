#Electromagnetic Particle-In-Cell Method
#For Computational Physics Reseach Purposes
#Primarily study and focus on plasma physics and nuclear fusion research
#Start from Non - Relativistic and Relativistic Lorentz Force Mechanics
#Then progressively develop in much further theory application like Lattice Field Theory,
#Quantum Mechanics, Relativistic Quantum Mechanics, QuantumFT, QuantumED, QuantumCD and etc.
#Developer and Researcher: Christopher Emmanuelle J. Visperas

import numpy as np

class Plane:
    def __init__(self, n : int, xi : float, xf : float):
        #number of dx, dy, dz
        self.n = n

        #initial x, y coordinate
        self.xa = xi[0]
        self.ya = xi[1]

        #final x, y coordinate
        self.xb = xf[0]
        self.yb = xf[1]
        
        self.Lx = self.xb - self.xa
        self.Ly = self.yb - self.ya
        
        self.x = np.linspace(self.xa, self.xb, self.n)
        self.y = np.linspace(self.ya, self.yb, self.n)

    def cartesian(self):
        return self.x, self.y
        
    def polar(self):
        r = np.sqrt(np.power(self.x, 2) + np.power(self.y, 2) + np.power(self.z, 2))
        theta = np.arctan2(self.y, self.x)
        return r, theta
        
class Solid:
    def __init__(self, n : int, xi : float, xf : float):
        #number of dx, dy, dz
        self.n = n
            
        #initial x, y, z coordinate
        self.xa = xi[0]
        self.ya = xi[1]
        self.za = xi[2]

        #final x, y, z coordinate
        self.xb = xf[0]
        self.yb = xf[1]
        self.zb = xf[2]
            
        self.x = np.linspace(self.xa, self.xb, self.n)
        self.y = np.linspace(self.ya, self.yb, self.n)
        self.z = np.linspace(self.za, self.zb, self.n)

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
