#Electromagnetic Particle-In-Cell Method
#Developer and Researcher: Christopher Emmanuelle J. Visperas

import numpy as np

# constant
epsilon_0 = 8.854187817e-12 #permittivity of free space
k0 = 1/(4*np.pi*epsilon_0)
mu_0 = 1.256637061e-6 # permeability of free space

#--------------------------------------------START EM PARTICLE-IN-CELL METHOD CLASSES--------------------------------------------
#-------------------------START FIELDS------------------------
class EMField:

    def __init__(self, xi, xf):

        #constant
        self.k0 = k0
        self.u0 = mu_0
        
        #initial x, y, z coordinate
        self.xi = xi[0]
        self.yi = xi[1]
        self.zi = xi[2]

        #final x, y, z coordinate
        self.xf = xf[0]
        self.yf = xf[1]
        self.zf = xf[2]

        #radial coordinate
        self.r = np.sqrt(np.power(self.xf - self.xi,2) + np.power(self.yf - self.yi,2) + np.power(self.zf - self.zi,2))

    #-------------------------------------------Field-------------------------------------------
    #Electric Field
    def E(self, q):
        E_x = (self.k0*q*(self.xf - self.xi))/np.power(self.r, 3)
        E_y = (self.k0*q*(self.yf - self.yi))/np.power(self.r, 3)
        E_z = (self.k0*q*(self.zf - self.zi))/np.power(self.r, 3)
        return E_x, E_y, E_z
    
    #Electric Potential
    def V(self, q):
        V_x = (k0*q*(self.xf - self.xi))/np.power(self.r,2)
        V_y = (k0*q*(self.yf - self.yi))/np.power(self.r,2)
        V_z = (k0*q*(self.zf - self.zi))/np.power(self.r,2)
        return V_x, V_y, V_z

    #External Magnetic Field
    def B2(self, I0, N):
        B_x2 = (mu_0*I0*N*(self.xf - self.xi))/(2*np.pi*np.power(self.r, 3))
        B_y2 = (mu_0*I0*N*(self.yf - self.yi))/(2*np.pi*np.power(self.r, 3))
        B_z2 = (mu_0*I0*N*(self.zf - self.zi))/(2*np.pi*np.power(self.r, 3))
        return B_x2, B_y2, B_z2

    #Radial Magnetic Field
    def B1(self, I0):
        B_x1 = (self.u0*I0*(self.xf - self.xi))/(2*np.pi*np.power(self.r, 3))
        B_y1 = -(self.u0*I0*(self.yf - self.yi))/(2*np.pi*np.power(self.r, 3))
        B_z1 = np.zeros_like(self.r)
        return B_x1, B_y1, B_z1

    #Particle's Electric Field
    def Ep(self, q1, q2):
        Ep_x = (self.k0*(q1*self.xi + q2*self.xf))/np.power(self.r, 3)
        Ep_y = (self.k0*(q1*self.yi + q2*self.yf))/np.power(self.r, 3)
        Ep_z = (self.k0*(q1*self.zi + q2*self.zf))/np.power(self.r, 3)
        return Ep_x, Ep_y, Ep_z

    #Particle's Magnetic Field
    def Bp(self, q1, q2, v1, v2):
        Bp_x = (self.u0/(4*np.pi*np.power(self.r, 2)))*(((q1*v1[1]*self.zi) - (q1*v1[2]*self.yi)) + ((q2*v2[1]*self.zf) - (q2*v2[2]*self.yf)))
        Bp_y = (self.u0/(4*np.pi*np.power(self.r, 2)))*(((q1*v1[2]*self.xi) - (q1*v1[0]*self.zi)) + ((q2*v2[2]*self.xf) - (q2*v2[0]*self.zf)))
        Bp_z = (self.u0/(4*np.pi*np.power(self.r, 2)))*(((q1*v1[0]*self.yi) - (q1*v1[1]*self.xi)) + ((q2*v2[0]*self.yf) - (q2*v2[1]*self.xf)))
        return Bp_x, Bp_y, Bp_z

    #Particle's Electric Potential Energy
    def Up(self, q1, q2):
        Up_x = (self.k0*q1*q2*(self.xf - self.xi))/np.power(self.r, 2)
        Up_y = (self.k0*q1*q2*(self.yf - self.yi))/np.power(self.r, 2)
        Up_z = (2*self.k0*q1*q2*(self.zf - self.zi))/np.power(self.r, 2)
        Up_d = (self.k0*q1*q2)/np.power(self.r, 2)
        return Up_d, Up_x, Up_y, Up_z

    #-------------------------------------------Particle-Grid Interaction-------------------------------------------
    #Electric Potential Energy
    def U(self, q1, q2):
        U_x = (self.k0*q1*q2*(self.xf - self.xi))/np.power(self.r, 2)
        U_y = (self.k0*q1*q2*(self.yf - self.yi))/np.power(self.r, 2)
        U_z = (2*self.k0*q1*q2*(self.zf - self.zi))/np.power(self.r, 2)
        return U_x, U_y, U_z

    #-------------------------------------------END-------------------------------------------

#---------------------------END FIELDS--------------------------