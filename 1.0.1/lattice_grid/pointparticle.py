#Electromagnetic Particle-In-Cell Method
#Developer and Researcher: Christopher Emmanuelle J. Visperas

import numpy as np

#-------------------------START PARTICLE------------------------
class particle:

    def __init__(self, m : float, q : float, u : float, E : float, B : float, t : float):

        #particle property
        self.m = m #kilogram
        self.c = 299792458 # m/s, speed of light

        #particle's charge grid
        self.q = q

        #particle's final velocity
        self.vx = u[0]
        self.vy = u[1]
        self.vz = u[2]

        #Electric Field from System
        self.Ex = E[0]
        self.Ey = E[1]
        self.Ez = E[2]

        #Magnetic Field from System
        self.Bx = B[0]
        self.By = B[1]
        self.Bz = B[2]

        #resultant velocity
        self.v = np.sqrt(np.power(self.vx,2) + np.power(self.vy,2) + np.power(self.vz,2))

        #time coordinate
        self.t = t
        
    def acceleration(self):
        
        #acceleration of the particle in electrostatic force between electrode surface area and charged particle
        a0_x = ((self.q/self.m)*((self.Ex + (((self.vy*self.Bz) - ((self.vz*self.By)))))))
        a0_y = ((self.q/self.m)*((self.Ey + (((self.vz*self.Bx) - ((self.vx*self.Bz)))))))
        a0_z = ((self.q/self.m)*((self.Ez + (((self.vx*self.By) - ((self.vy*self.Bx)))))))

        a_x = a0_x*np.sqrt(np.power(1 - np.power(self.vx/self.c, 2),3))
        a_y = a0_y*np.sqrt(np.power(1 - np.power(self.vy/self.c, 2),3))
        a_z = a0_z*np.sqrt(np.power(1 - np.power(self.vz/self.c, 2),3))
        
        return a_x, a_y, a_z

    def velocity(self, a_particle):
        #velocity of the particle in electrostatic force between electrode surface area and charged particle
        a_resultant = np.sqrt(np.power(a_particle[0],2) + np.power(a_particle[1],2) + np.power(a_particle[2],2))
        
        v_x = self.vx + ((a_particle[0]*self.t)/np.sqrt(1 - np.power(self.vx/self.c, 2)))
        v_y = self.vy + ((a_particle[1]*self.t)/np.sqrt(1 - np.power(self.vy/self.c, 2)))
        v_z = self.vz + ((a_particle[2]*self.t)/np.sqrt(1 - np.power(self.vz/self.c, 2)))
        return v_x, v_y, v_z

    def position(self, x):
        #position of the particle
        position_x = x[0] + ((self.vx*self.t)/np.sqrt(1 - np.power(self.vx/self.c, 2)))
        position_y = x[1] + ((self.vy*self.t)/np.sqrt(1 - np.power(self.vy/self.c, 2)))
        position_z = x[2] + ((self.vz*self.t)/np.sqrt(1 - np.power(self.vz/self.c, 2)))
        return position_x, position_y, position_z
#--------------------------END PARTICLE-------------------------
        
