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
dt = 1e-12# time step
delay = 6.25e-8
steps = int(np.ceil(60/dt))
tprint = 0.0
t_max = 60          # Maximum simulation time

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

#----------------------------------Particle - Field Initialization----------------------------------
#Charge Grid
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

#Electric potential of particle's interaction between them
Uqe_x = np.zeros(num_particles[0])
Uqe_y = np.zeros(num_particles[0])
Uqe_z = np.zeros(num_particles[0])
U_qe = np.zeros(num_particles[0])

Uqp_x = np.zeros(num_particles[1])
Uqp_y = np.zeros(num_particles[1])
Uqp_z = np.zeros(num_particles[1])
U_qp = np.zeros(num_particles[1])

#field weights
we = np.zeros((2, num_particles[0])) #for electron
wp = np.zeros((2, num_particles[1])) #for proton

#----------------------------------Particle - Particle Initialization----------------------------------
#storing velocity of electron interaction in otehr particles
vx_e = np.random.uniform(-c0, c0, num_particles[0])
vy_e = np.random.uniform(-c0, c0, num_particles[0])
vz_e = np.random.uniform(-c0, c0, num_particles[0])

#initial velocity of proton interaction in otehr particles
vx_p = np.random.uniform(-c0, c0, num_particles[1])
vy_p = np.random.uniform(-c0, c0, num_particles[1])
vz_p = np.random.uniform(-c0, c0, num_particles[1])

# Electric field for electron-electron
Eee_xField = np.zeros((num_particles[0], num_particles[0]))
Eee_yField = np.zeros((num_particles[0], num_particles[0]))
Eee_zField = np.zeros((num_particles[0], num_particles[0]))

# Electric field for electron-proton
Ep_xField = np.zeros(num_particles[1])
Ep_yField = np.zeros(num_particles[1])
Ep_zField = np.zeros(num_particles[1])

Ee_xField = np.zeros(num_particles[0])
Ee_yField = np.zeros(num_particles[0])
Ee_zField = np.zeros(num_particles[0])

# Electric field for proton-proton
Epp_xField = np.zeros((num_particles[1], num_particles[1]))
Epp_yField = np.zeros((num_particles[1], num_particles[1]))
Epp_zField = np.zeros((num_particles[1], num_particles[1]))

# Magnetic field for electron-electron
Bee_xField = np.zeros((num_particles[0], num_particles[0]))
Bee_yField = np.zeros((num_particles[0], num_particles[0]))
Bee_zField = np.zeros((num_particles[0], num_particles[0]))

# Magnetic field for electron-proton
Bp_xField = np.zeros(num_particles[1])
Bp_yField = np.zeros(num_particles[1])
Bp_zField = np.zeros(num_particles[1])

Be_xField = np.zeros(num_particles[0])
Be_yField = np.zeros(num_particles[0])
Be_zField = np.zeros(num_particles[0])

# Magnetic field for proton-proton
Bpp_xField = np.zeros((num_particles[1], num_particles[1]))
Bpp_yField = np.zeros((num_particles[1], num_particles[1]))
Bpp_zField = np.zeros((num_particles[1], num_particles[1]))

# Electric Potential for electron - electron
Uee_x = np.zeros((num_particles[0], num_particles[0]))
Uee_y = np.zeros((num_particles[0], num_particles[0]))
Uee_z = np.zeros((num_particles[0], num_particles[0]))

# Electric Potential for electron - proton
Upe_x = np.zeros((num_particles[1], num_particles[0]))
Upe_y = np.zeros((num_particles[1], num_particles[0]))
Upe_z = np.zeros((num_particles[1], num_particles[0]))

# Electric Potential for proton - proton
Upp_x = np.zeros((num_particles[1], num_particles[1]))
Upp_y = np.zeros((num_particles[1], num_particles[1]))
Upp_z = np.zeros((num_particles[1], num_particles[1]))

# Total Electric Potential 
U_ee = np.zeros((num_particles[0], num_particles[0]))
U_pe = np.zeros((num_particles[1], num_particles[0]))
U_pp = np.zeros((num_particles[1], num_particles[1]))

#--------------------------------------------START EM PARTICLE-IN-CELL METHOD CLASSES--------------------------------------------
#-------------------------START FIELDS------------------------
class RelavMech:
    def __init__(self, x, u):
        self.c = 299792458 #m/s
        
        self.x = x[0]
        self.y = x[1]
        self.z = x[2]

        self.vx = u[0]
        self.vy = u[1]
        self.vz = u[2]
        
    def gamma(self):
        gamma_x = 1 / np.sqrt(1 - np.power(self.vx/self.c, 2))
        gamma_y = 1 / np.sqrt(1 - np.power(self.vy/self.c, 2))
        gamma_z = 1 / np.sqrt(1 - np.power(self.vz/self.c, 2))
        return gamma_x, gamma_y, gamma_z

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

    #-------------------------------------------Field-------------------------------------------
    #Charge field
    def Q_field(self, Vab):
        #Charge from Electric Potential
        Q_x = (Vab*np.power(self.r, 2))/(k0*(self.x - self.x0))
        Q_y = (Vab*np.power(self.r, 2))/(k0*(self.y - self.y0))
        Q_z = (Vab*np.power(self.r, 2))/(k0*(self.z - self.z0))
        return Q_x, Q_y, Q_z
    
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

    #External Magnetic Field
    def B2(self, I0):
        B_x2 = (mu_0*I0*n*(self.x - self.x0))/(2*np.pi*np.power(self.r, 3))
        B_y2 = (mu_0*I0*n*(self.y - self.y0))/(2*np.pi*np.power(self.r, 3))
        B_z2 = (mu_0*I0*n*(self.z - self.z0))/(2*np.pi*np.power(self.r, 3))
        return B_x2, B_y2, B_z2

    #Radial Magnetic Field
    def B1(self, I0):
        B_x1 = (mu_0*I0*(self.x - self.x0))/(2*np.pi*np.power(self.r, 3))
        B_y1 = -(mu_0*I0*(self.y - self.y0))/(2*np.pi*np.power(self.r, 3))
        B_z1 = np.zeros_like(self.r)
        return B_x1, B_y1, B_z1

    #Particle's Electric Field
    def Ep(self, q):
        Ep_x = (k0*q*(self.x - self.x0))/np.power(self.r, 3)
        Ep_y = (k0*q*(self.y - self.y0))/np.power(self.r, 3)
        Ep_z = (k0*q*(self.z - self.z0))/np.power(self.r, 3)
        return Ep_x, Ep_y, Ep_z

    #Particle's Magnetic Field
    def Bp(self, q, v):
        Bp_x = (mu_0/(4*np.pi*np.power(self.r, 2)))*((q*v[1]*(self.z - self.z0)) - (q*v[2]*(self.y - self.y0)))
        Bp_y = (mu_0/(4*np.pi*np.power(self.r, 2)))*((q*v[2]*(self.x - self.x0)) - (q*v[0]*(self.z - self.z0)))
        Bp_z = (mu_0/(4*np.pi*np.power(self.r, 2)))*((q*v[0]*(self.y - self.y0)) - (q*v[1]*(self.x - self.x0)))
        return Bp_x, Bp_y, Bp_z

    #Particle's Electric Potential Energy
    def Up(self, q1, q2):
        Up_x = (k0*q1*q2*(self.x - self.x0))/np.power(self.r, 2)
        Up_y = (k0*q1*q2*(self.y - self.y0))/np.power(self.r, 2)
        Up_z = (2*k0*q1*q2*(self.z - self.z0))/np.power(self.r, 2)
        return Up_x, Up_y, Up_z

    #-------------------------------------------Particle-Grid Interaction-------------------------------------------
    #Electric Potential Energy
    def U(self, q1, q2):
        U_x = (k0*q1*q2[0]*(self.x - self.x0))/np.power(self.r, 2)
        U_y = (k0*q1*q2[1]*(self.y - self.y0))/np.power(self.r, 2)
        U_z = (2*k0*q1*q2[2]*(self.z - self.z0))/np.power(self.r, 2)
        return U_x, U_y, U_z

    #-------------------------------------------END-------------------------------------------
    #weighting the field
    def weight(self, x):
        w_x = ((x[0]*(self.dj - x[1])*(self.dk - x[2]))/(self.di*self.dj*self.dk))
        w_y = ((x[1]*(self.di - x[0])*(self.dk - x[2]))/(self.di*self.dj*self.dk))
        w_z = ((x[2]*(self.di - x[0])*(self.dj - x[1]))/(self.di*self.dj*self.dk))
        return w_x, w_y, w_z

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

for Ne in range(num_particles[0]):
    #initial velocity of electron
    ve_x = np.random.uniform(-c0, c0, num_particles[0])
    ve_y = np.random.uniform(-c0, c0, num_particles[0])
    ve_z = np.random.uniform(-c0, c0, num_particles[0])
    v1 = np.sqrt(np.power(ve_x[Ne], 2) + np.power(ve_y[Ne], 2) + np.power(ve_z[Ne], 2))
    gamma1 = 1/np.sqrt(1 - np.power(v1/c0, 2)

    if (gamma1 == np.nan):
        ve_x[Ne] = np.random.uniformn

#initial velocity of proton
vp_x = np.random.uniform(-c0, c0, num_particles[1])
vp_y = np.random.uniform(-c0, c0, num_particles[1])
vp_z = np.random.uniform(-c0, c0, num_particles[1])

#storing velocity of electron interaction in otehr particles
vx_e = np.random.uniform(-c0, c0, num_particles[0])
vy_e = np.random.uniform(-c0, c0, num_particles[0])
vz_e = np.random.uniform(-c0, c0, num_particles[0])

#initial velocity of proton interaction in otehr particles
vx_p = np.random.uniform(-c0, c0, num_particles[1])
vy_p = np.random.uniform(-c0, c0, num_particles[1])
vz_p = np.random.uniform(-c0, c0, num_particles[1])

#starting point of calculation
start_time = time.time()

# Create the figure and axes
fig = plt.figure(dpi = 100)
axis = fig.add_subplot(projection='3d')

# Main simulation loop
t = 0.0

while t < (t_max + 1):
    
    axis.cla()
    
    for Ne in range(num_particles[0]):
        x = x_electron[Ne]
        y = y_electron[Ne]
        z = z_electron[Ne]

        Sa_electron_field = Field([0, 0, a], [x, y, z])
        Sb_electron_field = Field([0, 0, l1 - a], [x, y, z])

        Sa_electron_external_MagneticField = Field([0, 0, a], [x, y, z])
        Sb_electron_external_MagneticField = Field([0, 0, l2 - a], [x, y, z])
        
        Q_xa, Q_ya, Q_za = Sa_electron_field.Q_field(75000)
        Q_xb, Q_yb, Q_zb = Sb_electron_field.Q_field(-75000)

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

        Uqex1, Uqey1, Uqez1 = Sa_electron_field.U(-e, [Q_xa, Q_ya, Q_za])
        Uqex2, Uqey2, Uqez2 = Sb_electron_field.U(-e, [Q_xb, Q_yb, Q_zb])

        Uqe_x[Ne] = (Uqex1 - Uqex2)
        Uqe_y[Ne] = (Uqey1 - Uqey2)
        Uqe_z[Ne] = (Uqez1 - Uqez2)
        U_qe[Ne] = np.sqrt(np.power((Uqex1 - Uqex2),2) + np.power((Uqey1 - Uqey2),2) + np.power((Uqez1 - Uqez2),2))

        ve_x[Ne] = 0.5*(np.sqrt((2*np.abs(Uqex1 - Uqex2))/me) - ve_x[Ne])
        ve_y[Ne] = 0.5*(np.sqrt((2*np.abs(Uqey1 - Uqey2))/me) - ve_y[Ne])
        ve_z[Ne] = 0.5*(np.sqrt((2*np.abs(Uqez1 - Uqez2))/me) - ve_z[Ne])

        electron_particle = Particle(me, -e, [ve_x[Ne], ve_y[Ne], ve_z[Ne]],
                                     [Ex_field[0][Ne], Ey_field[0][Ne], Ez_field[0][Ne]],
                                     [Bx_field[0][Ne], By_field[0][Ne], Bz_field[0][Ne]],
                                     dt)
        ax, ay, az = electron_particle.acceleration()
        vi, vj, vk = electron_particle.velocity([ax, ay, az])
        xe, ye, ze = electron_particle.position([vi, vj, vk])

        ve_x[Ne] = vi
        ve_y[Ne] = vj
        ve_z[Ne] = vk
        
        x_electron[Ne] = xe
        y_electron[Ne] = ye
        z_electron[Ne] = ze

        # Periodic boundary conditions
        x_electron = np.mod(x_electron, x_electron + 2*b) - b
        y_electron = np.mod(y_electron, y_electron + 2*b) - b
        z_electron = np.mod(z_electron - a, l1 - 2*a) + a


    # Calculate electric field due to proton's charges and positions
    for Np in range(num_particles[1]):
        
        x = x_proton[Np]
        y = y_proton[Np]
        z = z_proton[Np]

        #Electric Field between Electron and Electrode
        Sa_proton_field = Field([0, 0, a], [x, y, z])
        Sb_proton_field = Field([0, 0, l1 - a], [x, y, z])

        Sa_proton_external_MagneticField = Field([0, 0, a], [x, y, z])
        Sb_proton_external_MagneticField = Field([0, 0, l2 - a], [x, y, z])

        Q_xa, Q_ya, Q_za = Sa_proton_field.Q_field(75000)
        Q_xb, Q_yb, Q_zb = Sb_proton_field.Q_field(-75000)

        Ex1, Ey1, Ez1 = Sa_proton_field.E([Q_xa, Q_ya, Q_za])
        Ex2, Ey2, Ez2 = Sb_proton_field.E([Q_xb, Q_yb, Q_zb])

        Ex_field[1][Np] = (Ex1 + Ex2)
        Ey_field[1][Np] = (Ey1 + Ey2)
        Ez_field[1][Np] = (Ez1 + Ez2)

        B1_x1, B1_y1, B1_z1 = Sa_proton_field.B1(I)
        B1_x2, B1_y2, B1_z2 = Sb_proton_field.B1(I)

        B2_x1, B2_y1, B2_z1 = Sa_proton_external_MagneticField.B2(I)
        B2_x2, B2_y2, B2_z2 = Sb_proton_external_MagneticField.B2(I)

        Bx_field[1][Np] = ((B1_x1 + B1_x2) + (B2_x1 - B2_x2))
        By_field[1][Np] = ((B1_y1 + B1_y2) + (B2_y1 - B2_y2))
        Bz_field[1][Np] = ((B1_z1 + B1_z2) + (B2_z1 - B2_z2))

        Uqpx1, Uqpy1, Uqpz1 = Sa_proton_field.U(e, [Q_xa, Q_ya, Q_za])
        Uqpx2, Uqpy2, Uqpz2 = Sb_proton_field.U(e, [Q_xb, Q_yb, Q_zb])

        Uqp_x[Np] = (Uqpx1 - Uqpx2)
        Uqp_y[Np] = (Uqpy1 - Uqpy2)
        Uqp_z[Np] = (Uqpz1 - Uqpz2)
        U_qp[Np] = np.sqrt(np.power((Uqpx1 - Uqpx2),2) + np.power((Uqpy1 - Uqpy2),2) + np.power((Uqpz1 - Uqpz2),2))

        vp_x[Np] = 0.5*(np.sqrt((2*np.abs(Uqpx1 - Uqpx2))/mp) - vp_x[Np])
        vp_y[Np] = 0.5*(np.sqrt((2*np.abs(Uqpy1 - Uqpy2))/mp) - vp_y[Np])
        vp_z[Np] = 0.5*(np.sqrt((2*np.abs(Uqpz1 - Uqpz2))/mp) - vp_z[Np])

        proton_particle = Particle(mp, e, [vp_x[Np], vp_y[Np], vp_z[Np]],
                                     [Ex_field[1][Np], Ey_field[1][Np], Ez_field[1][Np]],
                                     [Bx_field[1][Np], By_field[1][Np], Bz_field[1][Np]],
                                     dt)
        
        ax, ay, az = proton_particle.acceleration()
        vi, vj, vk = proton_particle.velocity([ax, ay, az])
        xp, yp, zp = proton_particle.position([x, y, z])

        vp_x[Np] = vi
        vp_y[Np] = vj
        vp_z[Np] = vk
        
        x_proton[Np] = xp
        y_proton[Np] = yp
        z_proton[Np] = zp

        # Periodic boundary conditions
        x_proton = np.mod(x_proton, x_proton + 2*b) - b
        y_proton = np.mod(y_proton, y_proton + 2*b) - b
        z_proton = np.mod(z_proton - a, l1 - 2*a) + a


    print(f"{ve_x, ve_y, ve_z}")
    #-----------------------------Particle - Field Interaction -----------------------------
    #----------------------------------------END--------------------------------------------
    #-----------------------------Particle - Particle Interaction -----------------------------

    #for electron - electron interaction
    for Nei in range(num_particles[0]):
        
        x1 = x_electron[Nei]
        y1 = y_electron[Nei]
        z1 = z_electron[Nei]

        for Nej in range(1, num_particles[0]):
            
            x2 = x_electron[Nej]
            y2 = y_electron[Nej]
            z2 = z_electron[Nej]

            if(not(Nei == Nej)):
                ee_interaction = Field([x1, y1, z1], [x2, y2, z2])
                Ueex, Ueey, Ueez = ee_interaction.Up(-e, -e)

                Uee_x[Nei][Nej] = Ueex
                Uee_y[Nei][Nej] = Ueey
                Uee_z[Nei][Nej] = Ueez
                U_ee[Nei][Nej] = np.sqrt(np.power(Ueex,2) + np.power(Ueey,2) + np.power(Ueez,2))

                vx_e[Nej] = 0.5*(np.sqrt((2*np.abs(Uee_x[Nei][Nej]))/me) - vx_e[Nej])
                vy_e[Nej] = 0.5*(np.sqrt((2*np.abs(Uee_y[Nei][Nej]))/me) - vy_e[Nej])
                vy_e[Nej] = 0.5*(np.sqrt((2*np.abs(Uee_z[Nei][Nej]))/me) - vz_e[Nej])

                Eee_x, Eee_y, Eee_z = ee_interaction.Ep(-e)
                Bee_x, Bee_y, Bee_z = ee_interaction.Bp(-e, [vx_e[Nej], vy_e[Nej], vz_e[Nej]])

                Eee_xField[Nei][Nej] = Eee_x
                Eee_yField[Nei][Nej] = Eee_y
                Eee_zField[Nei][Nej] = Eee_z

                Bee_xField[Nei][Nej] = Bee_x
                Bee_yField[Nei][Nej] = Bee_y
                Bee_zField[Nei][Nej] = Bee_z

                electron_particle = Particle(me, -e, [vx_e[Nej], vy_e[Nej], vz_e[Nej]],
                                             [Eee_xField[Nei][Nej], Eee_yField[Nei][Nej], Eee_zField[Nei][Nej]],
                                             [Bee_xField[Nei][Nej], Bee_yField[Nei][Nej], Bee_zField[Nei][Nej]],
                                             dt)

                ax, ay, az = electron_particle.acceleration()
                vi, vj, vk = electron_particle.velocity([ax, ay, az])
                xe, ye, ze = electron_particle.position([x2, y2, z2])

                ve_x[Nej] = vi
                ve_y[Nej] = vj
                ve_z[Nej] = vk

                x_electron[Nej] = xe
                y_electron[Nej] = ye
                z_electron[Nej] = ze

                # Periodic boundary conditions
                x_electron = np.mod(x_electron, x_electron + 2*b) - b
                y_electron = np.mod(y_electron, y_electron + 2*b) - b
                z_electron = np.mod(z_electron - a, l1 - 2*a) + a

            else:
                break

    print(f"for ee_interaction: {x_electron, y_electron, z_electron}")
    
    #for the proton - proton interaction
    for Npi in range(num_particles[1]):

        x1 = x_proton[Npi]
        y1 = y_proton[Npi]
        z1 = z_proton[Npi]

        for Npj in range(1, num_particles[1]):
            
            x2 = x_proton[Npj]
            y2 = y_proton[Npj]
            z2 = z_proton[Npj]

            if(not(Npi == Npj)):
                pp_interaction = Field([x1, y1, z1], [x2, y2, z2])
                Uppx, Uppy, Uppz = pp_interaction.Up(e, e)

                Upp_x[Npi][Npj] = Uppx
                Upp_y[Npi][Npj] = Uppy
                Upp_z[Npi][Npj] = Uppz
                U_pp[Npi][Npj] = np.sqrt(np.power(Uppx,2) + np.power(Uppy,2) + np.power(Uppz,2))

                vx_p[Npj] = 0.5*(np.sqrt((2*np.abs(Upp_x[Npi][Npj]))/mp) - vx_p[Npj])
                vy_p[Npj] = 0.5*(np.sqrt((2*np.abs(Upp_y[Npi][Npj]))/mp) - vy_p[Npj])
                vy_p[Npj] = 0.5*(np.sqrt((2*np.abs(Upp_z[Npi][Npj]))/mp) - vz_p[Npj])

                Epp_x, Epp_y, Epp_z = pp_interaction.Ep(e)
                Bpp_x, Bpp_y, Bpp_z = pp_interaction.Bp(e, [vx_p[Npj], vy_p[Npj], vz_p[Npj]])

                Epp_xField[Npi][Npj] = Epp_x
                Epp_yField[Npi][Npj] = Epp_y
                Epp_zField[Npi][Npj] = Epp_z

                Bpp_xField[Npi][Npj] = Bpp_x
                Bpp_yField[Npi][Npj] = Bpp_y
                Bpp_zField[Npi][Npj] = Bpp_z

                proton_particle = Particle(mp, e, [vx_p[Npj], vy_p[Npj], vz_p[Npj]],
                                           [Epp_xField[Npi][Npj], Epp_yField[Npi][Npj], Epp_zField[Npi][Npj]],
                                           [Bpp_xField[Npi][Npj], Bpp_yField[Npi][Npj], Bpp_zField[Npi][Npj]],
                                           dt)

                ax, ay, az = proton_particle.acceleration()
                vi, vj, vk = proton_particle.velocity([ax, ay, az])
                xp, yp, zp = proton_particle.position([x2, y2, z2])

                ve_x[Npi] = vi
                ve_y[Npi] = vj
                ve_z[Npi] = vk

                x_proton[Npi] = xp
                y_proton[Npi] = yp
                z_proton[Npi] = zp

                # Periodic boundary conditions
                x_proton = np.mod(x_proton, x_proton + 2*b) - b
                y_proton = np.mod(y_proton, y_proton + 2*b) - b
                z_proton = np.mod(z_proton - a, l1 - 2*a) + a

            else:
                break

    print(f"for pp_interaction: {x_proton, y_proton, z_proton}")
    
    #for the proton - electron interaction
    for Np in range(num_particles[1]):

        x1 = x_proton[Np]
        y1 = y_proton[Np]
        z1 = z_proton[Np]

        for Ne in range(num_particles[0]):

            x2 = x_electron[Ne]
            y2 = y_electron[Ne]
            z2 = z_electron[Ne]
            
            #Electric Potential Energy between proton and electron
            pe_interaction = Field([x1, y1, z1], [x2, y2, z2])
            Upex, Upey, Upez = pe_interaction.Up(-e, e)

            Upe_x[Np][Ne] = Upex
            Upe_y[Np][Ne] = Upey
            Upe_z[Np][Ne] = Upez
            U_pe[Np][Ne] = np.sqrt(np.power(Upex,2) + np.power(Upey,2) + np.power(Upez,2))

            vx_e[Ne] = 0.5*(np.sqrt((2*np.abs(Upe_x[Np][Ne]))/me) - vx_e[Ne])
            vy_e[Ne] = 0.5*(np.sqrt((2*np.abs(Upe_y[Np][Ne]))/me) - vy_e[Ne])
            vy_e[Ne] = 0.5*(np.sqrt((2*np.abs(Upe_z[Np][Ne]))/me) - vz_e[Ne])

            vx_p[Np] = 0.5*(np.sqrt((2*np.abs(Upe_x[Np][Ne]))/mp) - vx_p[Np])
            vy_p[Np] = 0.5*(np.sqrt((2*np.abs(Upe_y[Np][Ne]))/mp) - vy_p[Np])
            vy_p[Np] = 0.5*(np.sqrt((2*np.abs(Upe_z[Np][Ne]))/mp) - vz_p[Np])

            Ee_x, Ee_y, Ee_z = pe_interaction.Ep(-e)
            Be_x, Be_y, Be_z = pe_interaction.Bp(-e, [vx_e[Ne], vy_e[Ne], vy_e[Ne]])
            
            Ep_x, Ep_y, Ep_z = pe_interaction.Ep(e)
            Bp_x, Bp_y, Bp_z = pe_interaction.Bp(e, [vx_p[Np], vy_p[Np], vz_p[Np]])

            Ee_xField[Ne] = Ee_x
            Ee_yField[Ne] = Ee_y
            Ee_zField[Ne] = Ee_z
            
            Ep_xField[Np] = Ep_x
            Ep_yField[Np] = Ep_y
            Ep_zField[Np] = Ep_z

            Bp_xField[Np] = Bp_x
            Bp_yField[Np] = Bp_y
            Bp_zField[Np] = Bp_z

            Be_xField[Ne] = Be_x
            Be_yField[Ne] = Be_y
            Be_zField[Ne] = Be_z

            electron_particle = Particle(me, -e, [vx_e[Ne], vy_e[Ne], vz_e[Ne]],
                                         [Ep_xField[Np], Ep_yField[Np], Ep_zField[Np]],
                                         [Bp_xField[Np], Bp_yField[Np], Bp_zField[Np]],
                                         dt)
            proton_particle = Particle(mp, e, [vx_p[Np], vy_p[Np], vz_p[Np]],
                                         [Ee_xField[Ne], Ee_yField[Ne], Ee_zField[Ne]],
                                         [Be_xField[Ne], Be_yField[Ne], Be_zField[Ne]],
                                         dt)

            ae_x, ae_y, ae_z = electron_particle.acceleration()
            ap_x, ap_y, ap_z = proton_particle.acceleration()
            
            vex, vey, vez = electron_particle.velocity([ae_x, ae_y, ae_z])
            vpx, vpy, vpz = proton_particle.velocity([ap_x, ap_y, ap_z])

            xe, ye, ze = electron_particle.position([x1, y1, z1])
            xp, yp, zp = proton_particle.position([x2, y2, z2])

            ve_x[Ne] = vex
            ve_y[Ne] = vey
            ve_z[Ne] = vez

            vp_x[Np] = vpx
            vp_y[Np] = vpy
            vp_z[Np] = vpz

            x_electron[Ne] = xe
            y_electron[Ne] = ye
            z_electron[Ne] = ze

            x_proton[Np] = xp
            y_proton[Np] = yp
            z_proton[Np] = yp

            # Periodic boundary conditions
            x_electron = np.mod(x_electron, x_electron + 2*b) - b
            y_electron = np.mod(y_electron, y_electron + 2*b) - b
            z_electron = np.mod(z_electron - a, l1 - 2*a) + a

            x_proton = np.mod(x_proton, x_proton + 2*b) - b
            y_proton = np.mod(y_proton, y_proton + 2*b) - b
            z_proton = np.mod(z_proton - a, l1 - 2*a) + a

    print(f"for pe_interaction: {x_electron, y_electron, z_electron} and {x_proton, y_proton, z_proton}")
    if(tprint >= delay):
        print(U_qe[:])
        tprint = 0.0
    tprint +=dt
    
    axis.scatter(z_electron, y_electron, x_electron, s = 10, marker = 'o', color = "blue")
    axis.scatter(z_proton, y_proton, x_proton, s = 10, marker = 'o', color = "red")
    axis.set_xlim(0, l1)
    axis.set_ylim(-2*b, 2*b)
    axis.set_zlim(-2*b, 2*b)
    axis.set_xlabel("z - Position")
    axis.set_ylabel("y - Position")
    axis.set_zlabel("x - Position")
    axis.set_title("Particle Motion with Interaction")
    plt.pause(delay)
    
    # Increment time
    t += dt






end_time = time.time()
execution_time = end_time - start_time
print(f"The time finish: {execution_time}")
plt.show()
