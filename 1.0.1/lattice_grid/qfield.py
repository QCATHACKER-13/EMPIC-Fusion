#Electromagnetic Particle-In-Cell Method
#Developer and Researcher: Christopher Emmanuelle J. Visperas

import numpy as np

class quant:
    def __init__(self, xi, xf, nx, ny, nz):
        #initial x, y, z coordinate
        self.xi = xi[0]
        self.yi = xi[1]
        self.zi = xi[2]

        #final x, y, z coordinate
        self.xf = xf[0]
        self.yf = xf[1]
        self.zf = xf[2]
        
        #divide space
        self.Nx = nx
        self.Ny = ny
        self.Nz = nz

        self.x, self.y, self.z = np.meshgrid(
            np.linspace(self.xi, self.xf, self.Nx),
            np.linspace(self.yi, self.yf, self.Ny),
            np.linspace(self.zi, self.zf, self.Nz)
        )
    
    def freeparticle(self, k):
        psi_x = (1/np.sqrt(self.xf - self.xi))*np.exp(-k[0]*self.x)
        psi_y = (1/np.sqrt(self.yf - self.yi))*np.exp(-k[1]*self.y)
        psi_z = (1/np.sqrt(self.zf - self.zi))*np.exp(-k[2]*self.z)
        return psi_x, psi_y, psi_z