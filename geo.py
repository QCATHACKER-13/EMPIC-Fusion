#Electromagnetic Particle-In-Cell Method
#Developer and Researcher: Christopher Emmanuelle J. Visperas

import numpy as np

class Plane:
    def __init__(self, xi, xf, n):
        #divide space
        self.nx = n[0]
        self.ny = n[1]
            
        #initial x, y, z coordinate
        self.xa = xi[0]
        self.ya = xi[1]
        
        #final x, y, z coordinate
        self.xb = xf[0]
        self.yb = xf[1]
        
        self.Lx = self.xb - self.xa
        self.Ly = self.yb - self.ya
        
        self.x = np.linspace(self.xa, self.xb, self.nx)
        self.y = np.linspace(self.ya, self.yb, self.ny)

    def cartesian(self):
        return self.x, self.y
        
    def polar(self):
        r = np.sqrt(np.power(self.x, 2) + np.power(self.y, 2) + np.power(self.z, 2))
        theta = np.arctan2(self.y, self.x)
        return r, theta
            
    def meshgrid(self, grid_x, grid_y):
        x, y = np.meshgrid(grid_x, grid_y)
        return x, y
        
class Solid:
    def __init__(self, xi, xf, n):
        #divide space
        self.nx = n[0]
        self.ny = n[1]
        self.nz = n[2]
            
        #initial x, y, z coordinate
        self.xa = xi[0]
        self.ya = xi[1]
        self.za = xi[2]
            
        #final x, y, z coordinate
        self.xb = xf[0]
        self.yb = xf[1]
        self.zb = xf[2]

        self.Lx = self.xb - self.xa
        self.Ly = self.yb - self.ya
        self.Lz = self.zb - self.za
            
        self.x = np.linspace(self.xa, self.xb, self.nx)
        self.y = np.linspace(self.ya, self.yb, self.ny)
        self.z = np.linspace(self.za, self.zb, self.nz)

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
        
    def meshgrid(self, grid_x, grid_y, grid_z):
        x, y, z = np.meshgrid(grid_x, grid_y, grid_z)
        return x, y, z
