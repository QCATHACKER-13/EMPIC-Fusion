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

    def __init__(self, xi, xf, n):

        #constant
        self.k0 = k0
        self.u0 = mu_0
        
        #initial x, y, z coordinate
        self.x0 = xi[0]
        self.y0 = xi[1]
        self.z0 = xi[2]

        #final x, y, z coordinate
        self.x = xf[0]
        self.y = xf[1]
        self.z = xf[2]

        #radial coordinate
        self.r = np.sqrt(np.power(self.x - self.x0,2) + np.power(self.y - self.y0,2) + np.power(self.z - self.z0,2))

        self.di = self.x - self.x0
        self.dj = self.x - self.x0
        self.dk = self.x - self.x0

    #-------------------------------------------Field-------------------------------------------
    #Charge field
    def Q_field(self, Vab):
        #Charge from Electric Potential
        Q_x = (Vab*np.power(self.r, 2))/(k0*(self.x - self.x0))
        Q_y = (Vab*np.power(self.r, 2))/(k0*(self.y - self.y0))
        Q_z = (Vab*np.power(self.r, 2))/(k0*(self.z - self.z0))
        return Q_x, Q_y, Q_z

    #particle field charge
    def e_field(self, q):
        e_x = q*((np.abs(self.x - self.di))/self.di)
        e_y = q*((np.abs(self.y - self.dj))/self.dj)
        e_z = q*((np.abs(self.z - self.dk))/self.dk)
        return e_x, e_y, e_z
        
    
    #Electric Field
    def E(self, q):
        E_x = (self.k0*q[0]*(self.x - self.x0))/np.power(self.r, 3)
        E_y = (self.k0*q[1]*(self.y - self.y0))/np.power(self.r, 3)
        E_z = (self.k0*q[2]*(self.z - self.z0))/np.power(self.r, 3)
        return E_x, E_y, E_z
    
    #Electric Potential
    def V(self, q):
        V_x = (k0*q*(self.x - self.x0))/np.power(self.r,2)
        V_y = (k0*q*(self.y - self.y0))/np.power(self.r,2)
        V_z = (k0*q*(self.z - self.z0))/np.power(self.r,2)
        return V_x, V_y, V_z

    #External Magnetic Field
    def B2(self, I0, N):
        B_x2 = (mu_0*I0*N*(self.x - self.x0))/(2*np.pi*np.power(self.r, 3))
        B_y2 = (mu_0*I0*N*(self.y - self.y0))/(2*np.pi*np.power(self.r, 3))
        B_z2 = (mu_0*I0*N*(self.z - self.z0))/(2*np.pi*np.power(self.r, 3))
        return B_x2, B_y2, B_z2

    #Radial Magnetic Field
    def B1(self, I0):
        B_x1 = (self.u0*I0*(self.x - self.x0))/(2*np.pi*np.power(self.r, 3))
        B_y1 = -(self.u0*I0*(self.y - self.y0))/(2*np.pi*np.power(self.r, 3))
        B_z1 = np.zeros_like(self.r)
        return B_x1, B_y1, B_z1

    #Particle's Electric Field
    def Ep(self, q1, q2):
        Ep_x = (self.k0*(q1*self.x0 + q2*self.x))/np.power(self.r, 3)
        Ep_y = (self.k0*(q1*self.y0 + q2*self.y))/np.power(self.r, 3)
        Ep_z = (self.k0*(q1*self.z0 + q2*self.z))/np.power(self.r, 3)
        return Ep_x, Ep_y, Ep_z

    #Particle's Magnetic Field
    def Bp(self, q1, q2, v1, v2):
        Bp_x = (self.u0/(4*np.pi*np.power(self.r, 2)))*(((q1*v1[1]*self.z0) - (q1*v1[2]*self.y0)) + ((q2*v2[1]*self.z) - (q2*v2[2]*self.y)))
        Bp_y = (self.u0/(4*np.pi*np.power(self.r, 2)))*(((q1*v1[2]*self.x0) - (q1*v1[0]*self.z0)) + ((q2*v2[2]*self.x) - (q2*v2[0]*self.z)))
        Bp_z = (self.u0/(4*np.pi*np.power(self.r, 2)))*(((q1*v1[0]*self.y0) - (q1*v1[1]*self.x0)) + ((q2*v2[0]*self.y) - (q2*v2[1]*self.x)))
        return Bp_x, Bp_y, Bp_z

    #Particle's Electric Potential Energy
    def Up(self, q1, q2):
        Up_x = (self.k0*q1*q2*(self.x - self.x0))/np.power(self.r, 2)
        Up_y = (self.k0*q1*q2*(self.y - self.y0))/np.power(self.r, 2)
        Up_z = (2*self.k0*q1*q2*(self.z - self.z0))/np.power(self.r, 2)
        Up_d = (self.k0*q1*q2)/np.power(self.r, 2)
        return Up_d, Up_x, Up_y, Up_z

    #-------------------------------------------Particle-Grid Interaction-------------------------------------------
    #Electric Potential Energy
    def U(self, q1, q2):
        U_x = (self.k0*q1*q2[0]*(self.x - self.x0))/np.power(self.r, 2)
        U_y = (self.k0*q1*q2[1]*(self.y - self.y0))/np.power(self.r, 2)
        U_z = (2*self.k0*q1*q2[2]*(self.z - self.z0))/np.power(self.r, 2)
        return U_x, U_y, U_z

    #-------------------------------------------END-------------------------------------------

#---------------------------END FIELDS--------------------------