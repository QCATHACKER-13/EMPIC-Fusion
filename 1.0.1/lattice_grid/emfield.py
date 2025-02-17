#Electromagnetic Particle-In-Cell Method
#For Computational Physics Reseach Purposes
#Primarily study and focus on plasma physics and nuclear fusion research
#Start from Non - Relativistic and Relativistic Lorentz Force Mechanics
#Then progressively develop in much further theory application like Lattice Field Theory,
#Quantum Mechanics, Relativistic Quantum Mechanics, QuantumFT, QuantumED, QuantumCD and etc.
#Developer and Researcher: Christopher Emmanuelle J. Visperas

import numpy as np

# constant
epsilon_0 = 8.854187817e-12 #permittivity of free space
k0 = 1/(4*np.pi*epsilon_0)
mu_0 = 1.256637061e-6 # permeability of free space

#--------------------------------------------START EM PARTICLE-IN-CELL METHOD CLASSES--------------------------------------------
#-------------------------START FIELDS------------------------
class Field:

    def __init__(self, xi:float, xf:float):

        #constant
        self.k0 = k0
        self.u0 = mu_0
        
        #initial x, y, z coordinate
        self.xa = xi[0]
        self.ya = xi[1]
        self.za = xi[2]

        #final x, y, z coordinate
        self.xb = xf[0]
        self.yb = xf[1]
        self.zb = xf[2]

        #radial coordinate
        self.r = np.sqrt(np.power(self.xb - self.xa,2) + np.power(self.yb - self.ya,2) + np.power(self.zb - self.za,2))

    #-------------------------------------------Field-------------------------------------------
    #Electric Field
    def E(self, q):
        E_x = (self.k0*q*(self.xb - self.xa))/np.power(self.r, 3)
        E_y = (self.k0*q*(self.yb - self.ya))/np.power(self.r, 3)
        E_z = (self.k0*q*(self.zb - self.za))/np.power(self.r, 3)
        return E_x, E_y, E_z

    #External Magnetic Field
    def B_dipole(self, N, I0):
        B_x2 = (mu_0*I0*N*(self.xb - self.xa))/(2*np.pi*np.power(self.r, 3))
        B_y2 = (mu_0*I0*N*(self.yb - self.ya))/(2*np.pi*np.power(self.r, 3))
        B_z2 = (mu_0*I0*N*(self.zb - self.za))/(2*np.pi*np.power(self.r, 3))
        return B_x2, B_y2, B_z2

    #Radial Magnetic Field
    def B_induced(self, I0):
        B_x1 = (self.u0*I0*(self.xb - self.xa))/(2*np.pi*np.power(self.r, 3))
        B_y1 = -(self.u0*I0*(self.yb - self.ya))/(2*np.pi*np.power(self.r, 3))
        B_z1 = np.zeros_like(self.r)
        return B_x1, B_y1, B_z1

    #-------------------------------------------END-------------------------------------------

#---------------------------END FIELDS--------------------------


#---------------------------TEST CODE--------------------------
#a = [0, 0, 1]
#b = [0, 0, 2]

#x, y, z = np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000), np.linspace(0, 2, 1000)

#field_a = Field(a, [x, y, z])
#field_b = Field([x, y, z], b)

#print(field_a.E(1))
#print()
#print(field_b.E(-1))

#---------------------------TEST: PASSED--------------------------