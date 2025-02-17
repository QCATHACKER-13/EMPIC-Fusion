#Electromagnetic Fused Atom Using Experimental Approach
#Computational Side of the Thesis Study
#Leader: Christopher Emmanuelle Visperas
#Members: John Kenneth De Leon
#         Jay Zard Gardose
#         Angellyn Santos
#         Michael Tagabi

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# constant
epsilon_0 = 8.854187817e-12 #permittivity of free space
k0 = 1/(4*np.pi*epsilon_0)
e = 1.60217663e-19 # elementary charge
mu_0 = 1.256637061e-6 # permeability of free space
mp = 1.672621637e-27 # kg, mass of a proton
mn = 1.674927211e-27 # kg, mass of a neutron
me = 9.10938215e-31 # kg, mass of an electron
c0 = 299792458 # m/s, speed of light

# Parameters
a = 0.5e-2 #center gap distance between two elctrodes
b = 0.15e-2 #radii of the electrode
d = 2.54e-2 #diameter of the cylinder glass 1 inch
l1 = 3.175e-2 #lenght of the cylinder glass 1.25 inch
Q = 3.3e-6#1.32e-5 for 150000 #3.0e-6 for 75000 # Coulumb length=0.2, 
I = np.pi*Q*np.power(0.5e-2,2)
n = 47875.72074 #total of turns = no. of turns in lenght * no. of turns in radii
R = np.sqrt(np.power(0.0802,2) - np.power(0.038,2))#m, final radii of the electromagnet and initial radii of the electromagnet
l2 = 0.056 #m, lenght of the electromagnet
Vab = 150000
volume = np.pi*np.power(d/2,2)*l1
ve = np.sqrt((2*Vab*e)/me) #for electron velocity
vp = np.sqrt((2*Vab*e)/mp) #for proton velocity
gamma = [1/np.sqrt(1 - np.power(ve/c0,2)),1/np.sqrt(1 - np.power(vp/c0,2))]
ds = 10
dt = 1e-3# time step
steps = int(np.ceil(60/dt))
dt = 0.01            # Time step
t_max = 1.0          # Maximum simulation time

print("Simulation of Nuclear Fusion Reaction")
print(f"The simulation is begin with a total frames = {steps}")
print()

# parameters for the particles
num_particles = [10, 5]

# x,y,z electron initial position
x_electron = np.random.uniform(-b, b, num_particles[0])
y_electron = np.random.uniform(-b, b, num_particles[0])
z_electron = np.random.uniform(a, l1 - a, num_particles[0])

# x,y,z proton initial position
x_proton = np.random.uniform(-b, b, num_particles[1])
y_proton = np.random.uniform(-b, b, num_particles[1])
z_proton = np.random.uniform(a, l1 - a, num_particles[1])

#initial velocity of electron
ve_x = np.random.uniform(-c0, c0, num_particles[0])
ve_y = np.random.uniform(-c0, c0, num_particles[0])
ve_z = np.random.uniform(-c0, c0, num_particles[0])

#initial velocity of proton
vp_x = np.random.uniform(-c0, c0, num_particles[1])
vp_y = np.random.uniform(-c0, c0, num_particles[1])
vp_z = np.random.uniform(-c0, c0, num_particles[1])

#max pressure 760 mmHg delta_pressure = 25 mmHg 1 mmHg = 133.322 Pascal
# Number of grid points
Nx = 1000
Ny = 1000
Nz = 1000

#field weights
w = np.zeros((2, num_particles[0])) #for electron
w2_e = np.zeros((2, num_particles[1])) #for proton

# Electric field for electron
Ex_field = np.zeros((2, Nx))
Ey_field = np.zeros((2, Ny))
Ez_field = np.zeros((2, Nz))

# Magnetic field for 
Bx_field = np.zeros((2, Nx))
By_field = np.zeros((2, Ny))
Bz_field = np.zeros((2, Nz))

#field weights
we = np.zeros((2, num_particles[0])) #for electron
wp = np.zeros((2, num_particles[1])) #for proton

#Electric potential of particle's interaction between them
Uqe_x = np.zeros(num_particles[0])
Uqe_y = np.zeros(num_particles[0])
Uqe_z = np.zeros(num_particles[0])
U_qe = np.zeros(num_particles[0])

Uqp_x = np.zeros(num_particles[1])
Uqp_y = np.zeros(num_particles[1])
Uqp_z = np.zeros(num_particles[1])
U_qp = np.zeros(num_particles[1])

Uee_x = np.zeros((num_particles[0], num_particles[0]))
Uee_y = np.zeros((num_particles[0], num_particles[0]))
Uee_z = np.zeros((num_particles[0], num_particles[0]))

Upe_x = np.zeros((num_particles[0], num_particles[1]))
Upe_y = np.zeros((num_particles[0], num_particles[1]))
Upe_z = np.zeros((num_particles[0], num_particles[1]))

Upp_x = np.zeros((num_particles[1], num_particles[1]))
Upp_y = np.zeros((num_particles[1], num_particles[1]))
Upp_z = np.zeros((num_particles[1], num_particles[1]))

U_ee = np.zeros((num_particles[0], num_particles[0]))
U_pe = np.zeros((num_particles[0], num_particles[1]))
U_pp = np.zeros((num_particles[1], num_particles[1]))

#--------------------------------------------START EM PARTICLE-IN-CELL METHOD CLASSES--------------------------------------------
#-------------------------START FIELDS------------------------
class Field:

    def __init__(self, xi, xf):

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

        self.di = (self.x - self.x0)
        self.dj = (self.y - self.y0)
        self.dk = (self.z - self.z0)

    #Electric Field
    def E(self, q):
        E_x = (k0*q*(self.x - self.x0))/np.power(self.r, 3)
        E_y = (k0*q*(self.y - self.y0))/np.power(self.r, 3)
        E_z = (k0*q*(self.z - self.z0))/np.power(self.r, 3)
        return E_x, E_y, E_z

    #weighting the field
    def weight(self, x):
        w_x = ((x[0]*(self.dj - x[1])*(self.dk - x[2]))/(self.di*self.dj*self.dk))
        w_y = ((x[1]*(self.di - x[0])*(self.dk - x[2]))/(self.di*self.dj*self.dk))
        w_z = ((x[2]*(self.di - x[0])*(self.dj - x[1]))/(self.di*self.dj*self.dk))
        return w_x, w_y, w_z
        

    #Electric Potential
    def V(self, q):
        V_x = (k0*q*(self.x - self.x0))/np.power(self.r,2)
        V_y = (k0*q*(self.y - self.y0))/np.power(self.r,2)
        V_z = (k0*q*(self.z - self.z0))/np.power(self.r,2)
        return V_x, V_y, V_z

    #Electric Potential Energy
    def U(self, q1, q2):
        U_x = (k0*q1*q2*(self.x - self.x0))/np.power(self.r, 2)
        U_y = (k0*q1*q2*(self.y - self.y0))/np.power(self.r, 2)
        U_z = (2*k0*q1*q2*(self.z - self.z0))/np.power(self.r, 2)
        return U_x, U_y, U_z

    def B1(self, I0):
        B_x1 = (mu_0*I0*(self.x - self.x0))/(2*np.pi*np.power(self.r, 3))
        B_y1 = -(mu_0*I0*(self.y - self.y0))/(2*np.pi*np.power(self.r, 3))
        B_z1 = np.zeros_like(self.r)
        return B_x1, B_y1, B_z1

    def B2(self, I0):
        B_x2 = (mu_0*I0*n*(self.x - self.x0))/(2*np.pi*np.power(self.r, 3))
        B_y2 = (mu_0*I0*n*(self.y - self.y0))/(2*np.pi*np.power(self.r, 3))
        B_z2 = (mu_0*I0*n*(self.z - self.z0))/(2*np.pi*np.power(self.r, 3))
        return B_x2, B_y2, B_z2

#---------------------------END FIELDS--------------------------

#-------------------------START PARTICLE------------------------
class Particle:

    def __init__(self, m, q, u, E, B, t):

        #particle property
        self.m = m #kilogram
        self.q = q #Coulumb
        self.c = 299792458 #m/s

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

        #Lorentz variance
        self.gamma = (np.sqrt(1 - np.power(self.v/self.c, 2))/np.power(np.sqrt(1 - np.power(self.v/self.c, 2)),2))

        #time coordinate
        self.t = t
        
    def acceleration(self):
        #acceleration of the particle in electrostatic force between electrode surface area and charged particle
        a_x = ((self.q/(self.m*self.c))*(((self.Ex*self.c)/self.gamma) + ((((self.vy*self.c*self.Bz)/self.gamma) - ((self.vz*self.c*self.By)/self.gamma)))))
        a_y = ((self.q/(self.m*self.c))*(((self.Ey*self.c)/self.gamma) + ((((self.vz*self.c*self.Bx)/self.gamma) - ((self.vx*self.c*self.Bz)/self.gamma)))))
        a_z = ((self.q/(self.m*self.c))*(((self.Ez*self.c)/self.gamma) + ((((self.vx*self.c*self.By)/self.gamma) - ((self.vy*self.c*self.Bx)/self.gamma)))))
        return a_x, a_y, a_z

    def velocity(self, a_particle):
        #velocity of the particle in electrostatic force between electrode surface area and charged particle
        a_resultant = np.sqrt(np.power(a_particle[0],2) + np.power(a_particle[1],2) + np.power(a_particle[2],2))
        v_x = ((0.5*a_particle[0]*self.t)/self.gamma)
        v_y = ((0.5*a_particle[1]*self.t)/self.gamma)
        v_z = ((0.5*a_particle[2]*self.t)/self.gamma)
        return v_x, v_y, v_z

    def position(self, v):
        #position of the particle
        position_x = ((v[0]*self.t)/self.gamma)
        position_y = ((v[1]*self.t)/self.gamma)
        position_z = ((v[2]*self.t)/self.gamma)
        return position_x, position_y, position_z

    def L(self, A, V, U):
        V = np.sqrt(np.power(V[0],2) + np.power(V[1],2) + np.power(V[2],2))
        KE = (0.5*self.m*np.power(self.c, 2)) - (self.q*V) + (self.q*(self.vx*A[0] + self.vy*A[1] + self.vz*A[2]))
        Lagrange = KE - (U[0] + U[1] + U[2])
        return Lagrange

    def H(self, k, V, U):
        V = np.sqrt(np.power(V[0],2) + np.power(V[1],2) + np.power(V[2],2))
        KE = (self.m*np.power(self.c,2)) - ((np.power(h_bar,2)/(2*self.m))*(np.power(k[0],2) + np.power(k[1],2) + np.power(k[2],2))) - (self.q*V)
        Hamilthon = KE + U[0] + U[1] + U[2]
        return Hamilthon
#--------------------------END PARTICLE-------------------------

# Main simulation loop
t = 0.0
while t < t_max:
    # Lorentz factor
    gamma = 1.0 / np.sqrt(1 - (v_particles/c)**2)
    
    # Update particle positions using velocity
    x_particles += v_particles * dt
    
    # Update particle velocities using the electric field
    v_particles += (charge / (gamma * mass)) * E0 * dt
    
    # Periodic boundary conditions
    x_particles = np.mod(x_particles, L)
    
    # Increment time
    t += dt

# Plot the final particle positions
plt.scatter(x_particles, np.zeros(N_particles), marker='o', label='Particles')
plt.xlabel('Position')
plt.title('Relativistic Particle-in-Cell Simulation')
plt.legend()
plt.show()
