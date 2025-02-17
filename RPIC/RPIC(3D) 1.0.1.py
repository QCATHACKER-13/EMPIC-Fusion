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

    def __init__(self, m, q, u0, u, E, B, t):

        #particle property
        self.m = m #kilogram
        self.q = q #Coulumb
        self.c = 299792458 #m/s

        #particle's final velocity
        self.v0_x = u0[0]
        self.v0_y = u0[1]
        self.v0_z = u0[2]

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
        self.v = np.sqrt(np.power(self.vx - self.v0_x,2) + np.power(self.vy - self.v0_y,2) + np.power(self.vz - self.v0_z,2))

        #Lorentz variance
        self.gamma = 1/np.sqrt(1 - np.power(self.v/self.c, 2))

        #time coordinate
        self.t = t
        
    def acceleration(self):
        #acceleration of the particle in electrostatic force between electrode surface area and charged particle
        a_x = ((self.q/(self.m*self.c))*(((self.Ex*self.c)/self.gamma) + (((((self.vy + self.v0_y)*self.c*0.5*self.Bz)/self.gamma) - (((self.vz + self.v0_z)*self.c*0.5*self.By)/self.gamma)))))
        a_y = ((self.q/(self.m*self.c))*(((self.Ey*self.c)/self.gamma) + (((((self.vz + self.v0_z)*self.c*0.5*self.Bx)/self.gamma) - (((self.vx + self.v0_x)*self.c*0.5*self.Bz)/self.gamma)))))
        a_z = ((self.q/(self.m*self.c))*(((self.Ez*self.c)/self.gamma) + (((((self.vx + self.v0_x)*self.c*0.5*self.By)/self.gamma) - (((self.vy + self.v0_y)*self.c*0.5*self.Bx)/self.gamma)))))
        return a_x, a_y, a_z

    def velocity(self, a_particle):
        #velocity of the particle in electrostatic force between electrode surface area and charged particle
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

# Create the figure and axes
fig = plt.figure(dpi = 100)
axis = fig.add_subplot(projection='3d')

# Main simulation loop
for step in range(steps):
    
    # Clear electric field
    Ex_field[:,:] = 0.0
    Ey_field[:,:] = 0.0
    Ez_field[:,:] = 0.0

    # Clear magnetic field
    Bx_field[:,:] = 0.0
    By_field[:,:] = 0.0
    Bz_field[:,:] = 0.0

    #Clear electric potential energy
    Uee_x[:,:] = 0.0
    Uee_y[:,:] = 0.0
    Uee_z[:,:] = 0.0

    Upe_x[:,:] = 0.0
    Upe_y[:,:] = 0.0
    Upe_z[:,:] = 0.0

    Upp_x[:,:] = 0.0
    Upp_y[:,:] = 0.0
    Upp_z[:,:] = 0.0

    Uqe_x[:] = 0.0
    Uqe_y[:] = 0.0
    Uqe_z[:] = 0.0

    Uqp_x[:] = 0.0
    Uqp_y[:] = 0.0
    Uqp_z[:] = 0.0

    U_ee[:,:] = 0.0
    U_pe[:,:] = 0.0
    U_pp[:,:] = 0.0
    U_qe[:] = 0.0
    U_qp[:] = 0.0

    #time to process
    axis.cla()
    start_time = time.time()

    for Ne in range(num_particles[0]):
        x = (x_electron[Ne] + ((2*b)/Nx))
        y = (y_electron[Ne] + ((2*b)/Ny))
        z = (z_electron[Ne] + ((l1 - (2*a))/Nz))

        Sa_electron_field = Field([0, 0, a], [x, y, z])
        Sb_electron_field = Field([0, 0, l1 - a], [x, y, z])

        Sa_electron_external_MagneticField = Field([0, 0, a], [x, y, z])
        Sb_electron_external_MagneticField = Field([0, 0, l2 - a], [x, y, z])

        Ex1, Ey1, Ez1 = Sa_electron_field.E(Q)
        Ex2, Ey2, Ez2 = Sb_electron_field.E(-Q)

        w1_x1, w1_y1, w1_z1 = Sa_electron_field.weight([x_electron[Ne], y_electron[Ne], z_electron[Ne]])
        w1_x2, w1_y2, w1_z2 = Sb_electron_field.weight([x_electron[Ne], y_electron[Ne], z_electron[Ne]])

        w2_x1, w2_y1, w2_z1 = Sa_electron_external_MagneticField.weight([x_electron[Ne], y_electron[Ne], z_electron[Ne]])
        w2_x2, w2_y2, w2_z2 = Sb_electron_external_MagneticField.weight([x_electron[Ne], y_electron[Ne], z_electron[Ne]])

        Ex_field[0][Ne] = ((w1_x1*Ex1) + (w1_x2*Ex2))
        Ey_field[0][Ne] = ((w1_y1*Ey1) + (w1_y2*Ey2))
        Ez_field[0][Ne] = ((w1_z1*Ez1) + (w1_z2*Ez2))
        
        B1_x1, B1_y1, B1_z1 = Sa_electron_field.B1(I)
        B1_x2, B1_y2, B1_z2 = Sb_electron_field.B1(I)

        B2_x1, B2_y1, B2_z1 = Sa_electron_external_MagneticField.B2(I)
        B2_x2, B2_y2, B2_z2 = Sb_electron_external_MagneticField.B2(I)

        Bx_field[0][Ne] = (0.5*(((w1_x1*B1_x1) + (w1_x2*B1_x2)) + ((w2_x1*B2_x1) - (w2_x2*B2_x2))))
        By_field[0][Ne] = (0.5*(((w1_y1*B1_y1) + (w1_y2*B1_y2)) + ((w2_y1*B2_y1) - (w2_y2*B2_y2))))
        Bz_field[0][Ne] = (0.5*(((w1_z1*B1_z1) + (w1_z2*B1_z2)) + ((w2_z1*B2_z1) - (w2_z2*B2_z2))))

        Uqex1, Uqey1, Uqez1 = Sa_electron_field.U(Q, -e)
        Uqex2, Uqey2, Uqez2 = Sb_electron_field.U(-Q, -e)

        Uqe_x[Ne] = (Uqex1 - Uqex2)
        Uqe_y[Ne] = (Uqey1 - Uqey2)
        Uqe_z[Ne] = (Uqez1 - Uqez2)
        U_qe[Ne] = np.sqrt(np.power((Uqex1 - Uqex2),2) + np.power((Uqey1 - Uqey2),2) + np.power((Uqez1 + Uqez2),2))

        ve_x[Ne] = ((((0.5*(w1_x1 + w2_x1))*np.sqrt((2*np.abs(Uqex1))/me)) - ((0.5*(w1_x2 + w2_x2))*np.sqrt((2*np.abs(Uqex2))/me))) - ve_x[Ne])
        ve_y[Ne] = ((((0.5*(w1_y1 + w2_y1))*np.sqrt((2*np.abs(Uqey1))/me)) - ((0.5*(w1_y2 + w2_y2))*np.sqrt((2*np.abs(Uqey2))/me))) - ve_y[Ne])
        ve_z[Ne] = ((((0.5*(w1_z1 + w2_z1))*np.sqrt((2*np.abs(Uqez1))/me)) - ((0.5*(w1_z2 + w2_z2))*np.sqrt((2*np.abs(Uqez2))/me))) - ve_z[Ne])

        #ve_x[Ne] = (np.sqrt((2*np.abs(Uqex1 - Uqex2))/me) - ve_x[Ne])
        #ve_y[Ne] = (np.sqrt((2*np.abs(Uqey1 - Uqey2))/me) - ve_y[Ne])
        #ve_z[Ne] = (np.sqrt((2*np.abs(Uqez1 - Uqez2))/me) - ve_z[Ne])

    print(ve_x)
    
    for Ne in range(num_particles[0]):

        x = (x_electron[Ne] + ((2*b)/Nx))
        y = (y_electron[Ne] + ((2*b)/Ny))
        z = (z_electron[Ne] + ((l1 - (2*a))/Nz))

        Sa_electron_field = Field([0, 0, a], [x, y, z])
        Sb_electron_field = Field([0, 0, l1 - a], [x, y, z])

        Uqex1, Uqey1, Uqez1 = Sa_electron_field.U(Q, -e)
        Uqex2, Uqey2, Uqez2 = Sb_electron_field.U(-Q, -e)

        Uqe_x[Ne] = (Uqex1 - Uqex2)
        Uqe_y[Ne] = (Uqey1 - Uqey2)
        Uqe_z[Ne] = (Uqez1 - Uqez2)
        U_qe[Ne] = np.sqrt(np.power((Uqex1 - Uqex2),2) + np.power((Uqey1 - Uqey2),2) + np.power((Uqez1 + Uqez2),2))

        ve_x2 = (np.sqrt((2*np.abs(Uqex2))/me) - np.sqrt((2*np.abs(Uqex1))/me))
        ve_y2 = (np.sqrt((2*np.abs(Uqey2))/me) - np.sqrt((2*np.abs(Uqey1))/me))
        ve_z2 = (np.sqrt((2*np.abs(Uqez2))/me) - np.sqrt((2*np.abs(Uqez1))/me))
        
        electron_particle = Particle(me, -e, [ve_x[Ne], ve_y[Ne], ve_z[Ne]],
                                     [ve_x2, ve_y2, ve_z2],
                                     [Ex_field[0][Ne], Ey_field[0][Ne], Ez_field[0][Ne]],
                                     [Bx_field[0][Ne], By_field[0][Ne], Bz_field[0][Ne]],
                                     dt)

        ax, ay, az = electron_particle.acceleration()
        vx, vy, vz = electron_particle.velocity([ax, ay, az])
        xe, ye, ze = electron_particle.position([vx, vy, vz])

        ve_x[Ne] = (ve_x[Ne] + vx)/(1 + (ve_x[Ne]*vx)/np.power(c0, 2))
        ve_y[Ne] = (ve_y[Ne] + vy)/(1 + (ve_y[Ne]*vy)/np.power(c0, 2))
        ve_z[Ne] = (ve_z[Ne] + vx)/(1 + (ve_z[Ne]*vz)/np.power(c0, 2))
        
        x_electron[Ne] = xe
        y_electron[Ne] = ye
        z_electron[Ne] = ze
        
    # Periodic boundary conditions
    x_electron = np.mod(x_electron, x_electron + 2*b) - b
    y_electron = np.mod(y_electron, y_electron + 2*b) - b
    z_electron = np.mod(z_electron - a, l1 - 2*a) + a
    
    # Plot particle positions
    axis.scatter(z_electron, y_electron, x_electron, s = 10, marker = 'o', color = "blue")
    
    #X.append(x_particles)
    #Y.append(y_particles)
    #Z.append(z_particles) np.linspace(0, 1*np.power(10, 12))

    #-----------------------------------------START PRINTING THE FINAL RESULT------------------------------
    #print(f"Electron's Position({x_electron}, {y_electron}, {z_electron})")
    #print(f"Electron's Velocity({ve_x}, {ve_y}, {ve_z})")
    #print(f"Proton's Position({x_proton}, {y_proton}, {z_proton})")
    #print(f"Proton's Velocity({vp_x}, {vp_y}, {vp_z})")
    #print()
    #print(f"Electric Field at x-axis: {Ex_field[0]}")
    #print(f"Electron - Electron Electric Potential Energy(eV) is {U_ee}")
    #print(f"Electron - Proton Electric Potential Energy(eV) is {U_pe} eV")
    #print(f"Proton - Proton Electric Potential Energy(eV) is {U_pp} eV")
    #print()
    #print(f"Electron - Electron Electric Potential Energy(eV) is {U_qe}")
    #print(f"Electron - Proton Electric Potential Energy(eV) is {U_qp} eV")
    #print()
    #print()
    #-----------------------------------------END PRINTING THE FINAL RESULT------------------------------

    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{step} of {steps} steps")
    print(f"The time finish: {execution_time}")
    print()

    axis.set_xlim(0, l1)
    axis.set_ylim(-b, b)
    axis.set_zlim(-b, b)
    axis.set_xlabel("z - Position")
    axis.set_ylabel("y - Position")
    axis.set_zlabel("x - Position")
    axis.set_title("Particle Motion with Interaction")
    plt.pause(dt)


plt.show()
