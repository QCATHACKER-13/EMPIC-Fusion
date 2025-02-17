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
ve_x = np.zeros(num_particles[0])
ve_y = np.zeros(num_particles[0])
ve_z = np.zeros(num_particles[0])

#initial velocity of proton
vp_x = np.zeros(num_particles[1])
vp_y = np.zeros(num_particles[1])
vp_z = np.zeros(num_particles[1])

#max pressure 760 mmHg delta_pressure = 25 mmHg 1 mmHg = 133.322 Pascal
# Number of grid points
Nx = 1000
Ny = 1000
Nz = 1000

#field weights
w = np.zeros((2, num_particles[0])) #for electron
w2_e = np.zeros((2, num_particles[1])) #for proton

#Carge Grid
Qx = np.zeros((2, Nx))
Qy = np.zeros((2, Ny))
Qz = np.zeros((2, Nz))

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

        self.di = (2*b)/Nx
        self.dj = (2*b)/Ny
        self.dk = (l1 - (2*a))/Nz

    #Charge Grid
    def Q_grid(self, Vo):
        #Charge from Electric Potential
        q_x = (Vo*np.power(self.r, 2))/(self.x - self.x0)
        q_y = (Vo*np.power(self.r, 2))/(self.y - self.y0)
        q_z = (Vo*np.power(self.r, 2))/(self.z - self.z0)
        return q_x, q_y, q_z
        
        
    #Electric Field
    def E(self, q):
        E_x = (k0*q[0]*(self.x - self.x0))/np.power(self.r, 3)
        E_y = (k0*q[1]*(self.y - self.y0))/np.power(self.r, 3)
        E_z = (k0*q[2]*(self.z - self.z0))/np.power(self.r, 3)
        return E_x, E_y, E_z
    
    #Electric Potential
    def V(self, q):
        V_x = (k0*q*(self.x - self.x0))/np.power(self.r,2)
        V_y = (k0*q*(self.y - self.y0))/np.power(self.r,2)
        V_z = (k0*q*(self.z - self.z0))/np.power(self.r,2)
        return V_x, V_y, V_z

    #Electric Potential Energy
    def U(self, q1, q2):
        U_x = (k0*q1[0]*q2[0]*(self.x - self.x0))/np.power(self.r, 2)
        U_y = (k0*q1[1]*q2[1]*(self.y - self.y0))/np.power(self.r, 2)
        U_z = (2*k0*q1[2]*q2[2]*(self.z - self.z0))/np.power(self.r, 2)
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

        #time coordinate
        self.t = t
        
    def acceleration(self):
        
        #acceleration of the particle in electrostatic force between electrode surface area and charged particle
        a0_x = ((self.q/self.m)*((self.Ex + (((self.vy*self.Bz) - ((self.vz*self.By)))))))
        a0_y = ((self.q/self.m)*((self.Ey + (((self.vz*self.Bx) - ((self.vx*self.Bz)))))))
        a0_z = ((self.q/self.m)*((self.Ez + (((self.vx*self.By) - ((self.vy*self.Bx)))))))
        
        a_x = a0_x/np.power(np.sqrt(1 - np.power(self.v/self.c, 2)), 3)
        a_y = a0_y/np.power(np.sqrt(1 - np.power(self.v/self.c, 2)), 3)
        a_z = a0_z/np.power(np.sqrt(1 - np.power(self.v/self.c, 2)), 3)
        
        return a_x, a_y, a_z

    def velocity(self, a_particle):
        #velocity of the particle in electrostatic force between electrode surface area and charged particle
        
        gamma = 1 / np.sqrt(1 - np.power(self.v/self.c,2))
        
        v_x = (self.vx + (a_particle[0]*self.t))*gamma
        v_y = (self.vy + (a_particle[1]*self.t))*gamma
        v_z = (self.vz + (a_particle[2]*self.t))*gamma
        return v_x, v_y, v_z

    def position(self, v):
        gamma = 1 / np.sqrt(1 - np.power(self.v/self.c, 2))
        
        #position of the particle
        position_x = ((v[0]*self.t)*gamma)
        position_y = ((v[1]*self.t)*gamma)
        position_z = ((v[2]*self.t)*gamma)
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

    for Ne in range(num_particles[0]):
        x = x_electron[Ne]
        y = y_electron[Ne]
        z = z_electron[Ne]

        Sa_electron_field = Field([0, 0, a], [x, y, z])
        Sb_electron_field = Field([0, 0, l1 - a], [x, y, z])

        Sa_electron_external_MagneticField = Field([0, 0, a], [x, y, z])
        Sb_electron_external_MagneticField = Field([0, 0, l2 - a], [x, y, z])
        
        Q_xa, Q_ya, Q_za = Sa_electron_field.Q_grid(75000)
        Q_xb, Q_yb, Q_zb = Sb_electron_field.Q_grid(-75000)

        Ex1, Ey1, Ez1 = Sa_electron_field.E([Q_xa, Q_ya, Q_za])
        Ex2, Ey2, Ez2 = Sb_electron_field.E([Q_xb, Q_yb, Q_zb])

        Ex_field[0][Ne] = (Ex1 + Ex2)
        Ey_field[0][Ne] = (Ey1 + Ey2)
        Ez_field[0][Ne] = (Ez1 + Ez2)

        B1_x1, B1_y1, B1_z1 = Sa_electron_field.B1(I)
        B1_x2, B1_y2, B1_z2 = Sb_electron_field.B1(I)

        B2_x1, B2_y1, B2_z1 = Sa_electron_external_MagneticField.B2(I)
        B2_x2, B2_y2, B2_z2 = Sb_electron_external_MagneticField.B2(I)

        Bx_field[0][Ne] = ((B1_x1 + B1_x2) + (B2_x1 - B2_x2))
        By_field[0][Ne] = ((B1_y1 + B1_y2) + (B2_y1 - B2_y2))
        Bz_field[0][Ne] = ((B1_z1 + B1_z2) + (B2_z1 - B2_z2))

        Uqex1, Uqey1, Uqez1 = Sa_electron_field.U([-e, -e, -e], [Q_xa, Q_ya, Q_za])
        Uqex2, Uqey2, Uqez2 = Sb_electron_field.U([-e, -e, -e], [Q_xb, Q_yb, Q_zb])

        Uqe_x[Ne] = (Uqex1 - Uqex2)
        Uqe_y[Ne] = (Uqey1 - Uqey2)
        Uqe_z[Ne] = (Uqez1 - Uqez2)
        U_qe[Ne] = np.sqrt(np.power(Uqe_x[Ne],2) + np.power(Uqe_y[Ne],2) + np.power(Uqe_z[Ne],2))
        
        #ve_x[Ne] = (0.5*((np.sqrt((2*np.abs(Uqex1))/me)) + (np.sqrt((2*np.abs(Uqex2))/me))))/np.sqrt(1 - ((2*U_qe[Ne])/(me*np.power(c0,2))))
        #ve_y[Ne] = (0.5*((np.sqrt((2*np.abs(Uqey1))/me)) + (np.sqrt((2*np.abs(Uqey2))/me))))/np.sqrt(1 - ((2*U_qe[Ne])/(me*np.power(c0,2))))
        #ve_z[Ne] = (0.5*((np.sqrt((2*np.abs(Uqez1))/me)) + (np.sqrt((2*np.abs(Uqez2))/me))))/np.sqrt(1 - ((2*U_qe[Ne])/(me*np.power(c0,2))))

        
        #ve_x[Ne] = (0.5*((np.sqrt((2*np.abs(Uqex1))/me)) + (np.sqrt((2*np.abs(Uqex2))/me)))) + c0
        #ve_y[Ne] = (0.5*((np.sqrt((2*np.abs(Uqey1))/me)) + (np.sqrt((2*np.abs(Uqey2))/me)))) + c0
        #ve_z[Ne] = (0.5*((np.sqrt((2*np.abs(Uqez1))/me)) + (np.sqrt((2*np.abs(Uqez2))/me)))) + c0

        #ve_x[Ne] = ((0.5*((np.sqrt((2*np.abs(Uqex1))/me)) + (np.sqrt((2*np.abs(Uqex2))/me)))) - ve_x[Ne])
        #ve_y[Ne] = ((0.5*((np.sqrt((2*np.abs(Uqey1))/me)) + (np.sqrt((2*np.abs(Uqey2))/me)))) - ve_y[Ne])
        #ve_z[Ne] = ((0.5*((np.sqrt((2*np.abs(Uqez1))/me)) + (np.sqrt((2*np.abs(Uqez2))/me)))) - ve_z[Ne])
        
        #ve_x[Ne] = (np.sqrt((2*np.abs(Uqex1 - Uqex2))/me) - ve_x[Ne])
        #ve_y[Ne] = (np.sqrt((2*np.abs(Uqey1 - Uqey2))/me) - ve_y[Ne])
        #ve_z[Ne] = (np.sqrt((2*np.abs(Uqez1 - Uqez2))/me) - ve_z[Ne])

        ve_x[Ne] = np.sqrt((2*np.abs(Uqex1 - Uqex2))/me)/np.sqrt(1 - ((2*U_qe[Ne])/(me*np.power(c0,2))))
        ve_y[Ne] = np.sqrt((2*np.abs(Uqey1 - Uqey2))/me)/np.sqrt(1 - ((2*U_qe[Ne])/(me*np.power(c0,2))))
        ve_z[Ne] = np.sqrt((2*np.abs(Uqez1 - Uqez2))/me)/np.sqrt(1 - ((2*U_qe[Ne])/(me*np.power(c0,2))))
        
        #ve_x[Ne] = c0*(1 - (1/np.sqrt(1 - (((2*U_qe[Ne])/me)/np.power(c0,2)))))
        #ve_y[Ne] = c0*(1 - (1/np.sqrt(1 - (((2*U_qe[Ne])/me)/np.power(c0,2)))))
        #ve_z[Ne] = c0*(1 - (1/np.sqrt(1 - (((2*U_qe[Ne])/me)/np.power(c0,2)))))

        print(f"{(ve_x[Ne]), (ve_y[Ne]), (ve_z[Ne])}")
        #print(f"{Qx[0][Ne], Qy[0][Ne], Qz[0][Ne]}")

        #Relativistic Part
        electron_particle = Particle(me, -e, [ve_x[Ne], ve_y[Ne], ve_z[Ne]],
                                     [Ex_field[0][Ne], Ey_field[0][Ne], Ez_field[0][Ne]],
                                     [Bx_field[0][Ne], By_field[0][Ne], Bz_field[0][Ne]],
                                     dt)
        ax, ay, az = electron_particle.acceleration()
        vx, vy, vz = electron_particle.velocity([ax, ay, az])
        xe, ye, ze = electron_particle.position([vx, vy, vz])
        #print(f"{electron_particle.v/c0}")

        #print(f"{ax, ay, az}, {vx, vy, vz}, {xe, ye, ze}")
        
        x_electron[Ne] += xe
        y_electron[Ne] += ye
        z_electron[Ne] += ze




    
    # Increment time
    t += dt

# Plot the final particle positions
#plt.scatter(x_particles, np.zeros(Nx), marker='o', label='Particles')
#plt.xlabel('Position')
#plt.title('Relativistic Particle-in-Cell Simulation')
#plt.legend()
#plt.show()
