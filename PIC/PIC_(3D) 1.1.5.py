#Electromagnetic Fused Atom Using Experimental Approach
#Computational Side of the Thesis Study
#Leader: Christopher Emmanuelle Visperas
#Members: John Kenneth De Leon
#         Jay Zard Gardose
#         Angellyn Santos
#         Michael Tagabi

import time
import matplotlib.pyplot as plt
import numpy as np
from random import random

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
Q = 3.0e-11 #aCoulumb length=0.2, 
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
dt = 1e-5# time step
steps = int(np.ceil(60/dt))

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

#max pressure 760 mmHg delta_pressure = 25 mmHg 1 mmHg = 133.322 Pascal
# Number of grid points
Nx = 1000
Ny = 1000
Nz = 1000

#initial velocity of electron
ve_x = np.zeros(num_particles[0])
ve_y = np.zeros(num_particles[0])
ve_z = np.zeros(num_particles[0])

#initial velocity of proton
vp_x = np.zeros(num_particles[1])
vp_y = np.zeros(num_particles[1])
vp_z = np.zeros(num_particles[1])

# Electric field for electron
Ex_field = np.zeros((2, Nx))
Ey_field = np.zeros((2, Ny))
Ez_field = np.zeros((2, Nz))

# Magnetic field for 
Bx_field = np.zeros((2, Nx))
By_field = np.zeros((2, Ny))
Bz_field = np.zeros((2, Nz))

#Electric potential of particle's interaction between them
Uqe_x = np.zeros(num_particles[0])
Uqe_y = np.zeros(num_particles[0])
Uqe_z = np.zeros(num_particles[0])
U_qe = np.zeros(num_particles[0])

Uqp_x = np.zeros(num_particles[1])
Uqp_y = np.zeros(num_particles[1])
Uqp_z = np.zeros(num_particles[1])
U_qp = np.zeros(num_particles[1])

Uee_x = np.zeros((2, Nx))
Uee_y = np.zeros((2, Ny))
Uee_z = np.zeros((2, Nz))

Upe_x = np.zeros((2, Nx))
Upe_y = np.zeros((2, Ny))
Upe_z = np.zeros((2, Nz))

Upp_x = np.zeros((2, Nx))
Upp_y = np.zeros((2, Ny))
Upp_z = np.zeros((2, Nz))

U_ee = np.zeros((2, Nx))
U_pe = np.zeros((2, Ny))
U_pp = np.zeros((2, Nz))

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

    #Electric Field
    def E(self, q):
        E_x = (2*k0*q*(self.x - self.x0))/np.power(self.r, 3)
        E_y = (k0*q*(self.y - self.y0))/np.power(self.r, 3)
        E_z = (k0*q*(self.z - self.z0))/np.power(self.r, 3)
        return E_x, E_y, E_z

    #Electric Potential
    def V(self, q):
        V_x = (2*k0*q*(self.x - self.x0))/np.power(self.r,2)
        V_y = (k0*q*(self.y - self.y0))/np.power(self.r,2)
        V_z = (k0*q*(self.z - self.z0))/np.power(self.r,2)
        return V_x, V_y, V_z

    #Electric Potential Energy
    def U(self, q1, q2):
        U_x = (k0*q1*q2*(self.x - self.x0))/np.power(self.r, 3)
        U_y = (k0*q1*q2*(self.y - self.y0))/np.power(self.r, 3)
        U_z = (k0*q1*q2*(self.z - self.z0))/np.power(self.r, 3)
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
        self.gamma = 1/np.sqrt(1 - np.power(self.v/self.c, 2))

        #time coordinate
        self.t = t

    def acceleration(self):
        #acceleration of the particle in electrostatic force between electrode surface area and charged particle
        a_x = ((self.q/self.m)*(self.Ex + ((((self.vy/self.gamma)*self.Bz) - ((self.vz/self.gamma)*self.By)))))
        a_y = ((self.q/self.m)*(self.Ey + ((((self.vz/self.gamma)*self.Bx) - ((self.vx/self.gamma)*self.Bz)))))
        a_z = ((self.q/self.m)*(self.Ez + ((((self.vx /self.gamma)*self.By) - ((self.vy/self.gamma)*self.Bx)))))
        return a_x, a_y, a_z

    def velocity(self, a):
        #velocity of the particle in electrostatic force between electrode surface area and charged particle
        v_x = (0.5*a[0]*self.t)
        v_y = (0.5*a[1]*self.t)
        v_z = (0.5*a[2]*self.t)
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
#--------------------------------------------START EM PARTICLE-IN-CELL METHOD CLASSES--------------------------------------------

# Create the figure and axes
fig = plt.figure()
axis = fig.add_subplot(projection = "3d")


for step in range(100):
    
    # Clear electric field
    Ex_field[:,:] = 0.0
    Ey_field[:,:] = 0.0
    Ez_field[:,:] = 0.0

    # Clear magnetic field
    Bx_field[:,:] = 0.0
    By_field[:,:] = 0.0
    Bz_field[:,:] = 0.0

    #time of start to process
    start_time = time.time()
    axis.cla()

    #-------------------------------START PARTICLE - FIELD INTERACTION-------------------------------

    #------------Electron - Field Interaction-------------
    for Ne in range(num_particles[0]):

        x = (x_electron[Ne]/(b*Nx))
        y = (y_electron[Ne]/(b*Ny))
        z = (z_electron[Ne]/(l1*Nz))
        
        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(np.abs(z))

        if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):

            #Electric Field between Electron and Electrode
            Sa_electron_field = Field([0, 0, a], [x, y, z])
            Sb_electron_field = Field([0, 0, l1 - a], [x, y, z])

            Ex1, Ey1, Ez1 = Sa_electron_field.E(Q)
            Ex2, Ey2, Ez2 = Sb_electron_field.E(-Q)

            Ex_field[0][grid_x] = (Ex1 + Ex2)
            Ey_field[0][grid_y] = (Ey1 + Ey2)
            Ez_field[0][grid_z] = (Ez1 + Ez2)

            B1_x1, B1_y1, B1_z1 = Sa_electron_field.B1(I)
            B1_x2, B1_y2, B1_z2 = Sb_electron_field.B1(I)

            B2_x1, B2_y1, B2_z1 = Sa_electron_field.B2(I)
            B2_x2, B2_y2, B2_z2 = Sb_electron_field.B2(I)

            B1_x = (B1_x1 + B1_x2)
            B1_y = (B1_y1 + B1_y2)
            B1_z = (B1_z1 + B1_z2)

            B2_x = (B2_x1 + B2_x2)
            B2_y = (B2_y1 + B2_y2)
            B2_z = (B2_z1 + B2_z2)
            
            Bx_field[0][grid_x] = (0.5*(B1_x + B2_x))
            By_field[0][grid_y] = (0.5*(B1_y + B2_y))
            Bz_field[0][grid_z] = (0.5*(B1_z + B2_z))

            Uqex1, Uqey1, Uqez1 = Sa_electron_field.U(Q, -e)
            Uqex2, Uqey2, Uqez2 = Sb_electron_field.U(-Q, -e)

            Uqe_x[Ne] = (Uqex1 - Uqex2)
            Uqe_y[Ne] = (Uqey1 - Uqey2)
            Uqe_z[Ne] = (Uqez1 - Uqez2)
            U_qe[Ne] = np.sqrt(np.power((Uqex1 - Uqex2),2) + np.power((Uqey1 - Uqey2),2) + np.power((Uqez1 - Uqez2),2))
            
            ve_x[Ne] = (np.sqrt((2*np.abs(Uqex1))/me) - np.sqrt((2*np.abs(Uqex2))/me))
            ve_y[Ne] = (np.sqrt((2*np.abs(Uqey1))/me) - np.sqrt((2*np.abs(Uqey2))/me))
            ve_z[Ne] = (np.sqrt((2*np.abs(Uqez1))/me) - np.sqrt((2*np.abs(Uqez2))/me))

    for Ne in range(num_particles[0]):
        x = (x_electron[Ne]/(b*Nx))
        y = (y_electron[Ne]/(b*Ny))
        z = (z_electron[Ne]/(l1*Nz))

        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(np.abs(z))

        if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):
            electron_particle = Particle(me, -e, [ve_x[Ne], ve_y[Ne], ve_z[Ne]],
                                   [Ex_field[0][grid_x], Ey_field[0][grid_y], Ez_field[0][grid_z]],
                                   [Bx_field[0][grid_x], By_field[0][grid_y], Bz_field[0][grid_z]],
                                   dt)
            
            ax, ay, az = electron_particle.acceleration()
            vx, vy, vz = electron_particle.velocity([ax, ay, az])
            xe, ye, ze = electron_particle.position([vx, vy, vz])

            ve_x[Ne] = vx
            ve_y[Ne] = vy
            ve_z[Ne] = vz

            x_electron[Ne] = xe
            y_electron[Ne] = ye
            z_electron[Ne] = ze


    #------------Proton - Field Interaction-------------
    for Np in range(num_particles[1]):

        x = (x_proton[Np]/(b*Nx))
        y = (y_proton[Np]/(b*Ny))
        z = (z_proton[Np]/(l1*Nz))
        
        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(np.abs(z))

        if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):

            #Electric Field between Electron and Electrode
            Sa_proton_field = Field([0, 0, a], [x, y, z])
            Sb_proton_field = Field([0, 0, l1 - a], [x, y, z])

            Ex1, Ey1, Ez1 = Sa_proton_field.E(Q)
            Ex2, Ey2, Ez2 = Sb_proton_field.E(-Q)

            Ex_field[1][grid_x] = (Ex1 + Ex2)
            Ey_field[1][grid_y] = (Ey1 + Ey2)
            Ez_field[1][grid_z] = (Ez1 + Ez2)

            B1_x1, B1_y1, B1_z1 = Sa_proton_field.B1(I)
            B1_x2, B1_y2, B1_z2 = Sb_proton_field.B1(I)

            B2_x1, B2_y1, B2_z1 = Sa_proton_field.B2(I)
            B2_x2, B2_y2, B2_z2 = Sb_proton_field.B2(I)

            B1_x = (B1_x1 + B1_x2)
            B1_y = (B1_y1 + B1_y2)
            B1_z = (B1_z1 + B1_z2)

            B2_x = (B2_x1 + B2_x2)
            B2_y = (B2_y1 + B2_y2)
            B2_z = (B2_z1 + B2_z2)
            
            Bx_field[1][grid_x] = (0.5*(B1_x + B2_x))
            By_field[1][grid_y] = (0.5*(B1_y + B2_y))
            Bz_field[1][grid_z] = (0.5*(B1_z + B2_z))

            Uqpx1, Uqpy1, Uqpz1 = Sa_proton_field.U(Q, e)
            Uqpx2, Uqpy2, Uqpz2 = Sb_proton_field.U(-Q, e)

            Uqp_x[Np] = (Uqpx1 - Uqpx2)
            Uqp_y[Np] = (Uqpy1 - Uqpy2)
            Uqp_z[Np] = (Uqpz1 - Uqpz2)
            U_qp[Np] = np.sqrt(np.power((Uqpx1 - Uqpx2),2) + np.power((Uqpy1 - Uqpy2),2) + np.power((Uqpz1 - Uqpz2),2))
            
            vp_x[Np] = (np.sqrt((2*np.abs(Uqpx1))/mp) - np.sqrt((2*np.abs(Uqpx2))/mp))
            vp_y[Np] = (np.sqrt((2*np.abs(Uqpy1))/mp) - np.sqrt((2*np.abs(Uqpy2))/mp))
            vp_z[Np] = (np.sqrt((2*np.abs(Uqpz1))/mp) - np.sqrt((2*np.abs(Uqpz2))/mp))

    for Np in range(num_particles[1]):
        x = (x_proton[Np]/(b*Nx))
        y = (y_proton[Np]/(b*Ny))
        z = (z_proton[Np]/(l1*Nz))

        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(np.abs(z))

        if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):
            proton_particle = Particle(mp, e, [vp_x[Np], vp_y[Np], vp_z[Np]],
                                   [Ex_field[1][grid_x], Ey_field[1][grid_y], Ez_field[1][grid_z]],
                                   [Bx_field[1][grid_x], By_field[1][grid_y], Bz_field[1][grid_z]],
                                   dt)
            
            ax, ay, az = proton_particle.acceleration()
            vx, vy, vz = proton_particle.velocity([ax, ay, az])
            xp, yp, zp = proton_particle.position([vx, vy, vz])

            vp_x[Np] = vx
            vp_y[Np] = vy
            vp_z[Np] = vz

            x_proton[Np] = xp
            y_proton[Np] = yp
            z_proton[Np] = zp

    x_electron = np.mod(x_electron, x_electron + (2*b)) - b
    y_electron = np.mod(y_electron, y_electron + (2*b)) - b
    z_electron = np.mod(z_electron - a, l1 - (2*a)) + a
    print(Uqe_x)

    axis.set_xlim(0, l1)
    axis.set_ylim(-b, b)
    axis.set_zlim(-b, b)
    axis.set_xlabel("Z - Position")
    axis.set_ylabel("Y - Position")
    axis.set_zlabel("X - Position")
    plt.pause(0.001)

plt.show()
