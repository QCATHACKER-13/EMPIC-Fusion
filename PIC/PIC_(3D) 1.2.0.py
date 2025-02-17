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

    #Electric Field
    def E(self, q):
        E_x = (k0*q*(self.x - self.x0))/np.power(self.r, 3)
        E_y = (k0*q*(self.y - self.y0))/np.power(self.r, 3)
        E_z = (k0*q*(self.z - self.z0))/np.power(self.r, 3)
        return E_x, E_y, E_z

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
        self.gamma = 1/np.sqrt(1 - np.power(self.v/self.c, 2))

        #time coordinate
        self.t = t
        
    def acceleration(self):
        #acceleration of the particle in electrostatic force between electrode surface area and charged particle
        a_x = ((self.q/(self.m*self.c))*(((self.Ex*self.c)/self.gamma) + ((((self.vy*self.Bz)/self.gamma) - ((self.vz*self.By)/self.gamma)))))
        a_y = ((self.q/(self.m*self.c))*(((self.Ey*self.c)/self.gamma) + ((((self.vz*self.Bx)/self.gamma) - ((self.vx*self.Bz)/self.gamma)))))
        a_z = ((self.q/(self.m*self.c))*(((self.Ez*self.c)/self.gamma) + ((((self.vx*self.By)/self.gamma) - ((self.vy*self.Bx)/self.gamma)))))
        return a_x, a_y, a_z

    def velocity(self, a):
        #velocity of the particle in electrostatic force between electrode surface area and charged particle
        v_x = ((self.c - (0.5*a[0]*self.t))/self.gamma)
        v_y = ((self.c - (0.5*a[1]*self.t))/self.gamma)
        v_z = ((self.c - (0.5*a[2]*self.t))/self.gamma)
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

# x,y electron initial position
x_electron = np.random.uniform(-b, b, num_particles[0])
y_electron = np.random.uniform(-b, b, num_particles[0])
z_electron = np.random.uniform(a, l1 - a, num_particles[0])

# x,y proton initial position
x_proton = np.random.uniform(-b, b, num_particles[1])
y_proton = np.random.uniform(-b, b, num_particles[1])
z_proton = np.random.uniform(a, l1 - a, num_particles[1])

#----------------------------------------------------End Particle System------------------------------------------------

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
    
    #-----------------------------Particle - Particle Interaction -----------------------------

    #for electron - electron interaction
    for Nei in range(num_particles[0]):
        
        x1 = (x_electron[Nei] + ((l1 - 2*a)/(b * Nx)))
        y1 = (y_electron[Nei] + ((2*b)/Ny))
        z1 = (z_electron[Nei] + ((2*b)/Ny))
        
        #Electric Field between Electron and Electrode
        Sa_electron_field = Field([0, 0, a], [x1, y1, z1])
        Sb_electron_field = Field([0, 0, l1 - a], [x1, y1, z1])

        Exie1, Eyie1, Ezie1 = Sa_electron_field.E(Q)
        Exie2, Eyie2, Ezie2 = Sb_electron_field.E(-Q)

        Ex_field[0][Nei] = (Exie1 + Exie2)
        Ey_field[0][Nei] = (Eyie1 + Eyie2)
        Ez_field[0][Nei] = (Ezie1 + Ezie2)

        Bxie1_1, Byie1_1, Bzie1_1 = Sa_electron_field.B1(I)
        Bxie2_1, Byie2_1, Bzie2_1 = Sb_electron_field.B1(I)

        Bxie1_2, Byie1_2, Bzie1_2 = Sa_electron_field.B2(I)
        Bxie2_2, Byie2_2, Bzie2_2 = Sb_electron_field.B2(I)

        Bx_field[0][Nei] = (0.5*((Bxie1_1 + Bxie2_1) + (Bxie1_2 - Bxie2_2)))
        By_field[0][Nei] = (0.5*((Byie1_1 + Byie2_1) + (Byie1_2 - Byie2_2)))
        Bz_field[0][Nei] = (0.5*((Bzie1_1 + Bzie2_1) + (Bzie1_2 - Bzie2_2)))

        Uqex1, Uqey1, Uqez1 = Sa_electron_field.U(Q, -e)
        Uqex2, Uqey2, Uqez2 = Sb_electron_field.U(-Q, -e)

        Uqe_x[Nei] = (Uqex1 + Uqex1)
        Uqe_y[Nei] = (Uqey1 + Uqey1)
        Uqe_z[Nei] = (Uqez1 + Uqez1)
        U_qe[Nei] = np.sqrt(np.power((Uqex2 + Uqex1),2) + np.power((Uqey2 + Uqey1),2) + np.power((Uqez2 + Uqez1),2))

        ve_x[Nei] = (0.5*(np.sqrt((2*np.abs(Uqex2))/me) - np.sqrt((2*np.abs(Uqex1))/me)))
        ve_y[Nei] = (0.5*(np.sqrt((2*np.abs(Uqey2))/me) - np.sqrt((2*np.abs(Uqey1))/me)))
        ve_z[Nei] = (0.5*(np.sqrt((2*np.abs(Uqez2))/me) - np.sqrt((2*np.abs(Uqez1))/me)))

        for Nej in range(num_particles[0]):

            x2 = (x_electron[Nej] + ((l1 - 2*a)/(b * Nx)))
            y2 = (y_electron[Nej] + ((2*b)/Ny))
            z2 = (z_electron[Nej] + ((2*b)/Ny))

            if not(Nei == Nej):

                ee_interaction = Field([x1, y1, z1], [x2, y2, z2])
                Ueex, Ueey, Ueez = ee_interaction.U(-e, -e)

                Uee_x[Nei][Nej] = Ueex
                Uee_y[Nei][Nej] = Ueey
                Uee_z[Nei][Nej] = Ueez
                U_ee[Nei][Nej] = np.sqrt(np.power(Ueex,2) + np.power(Ueey,2) + np.power(Ueez,2))

                electron_particle = Particle(me, -e, [(ve_x[Nei] + ve_x[Nej])/2, (ve_y[Nei] + ve_y[Nej])/2, (ve_z[Nei] + ve_z[Nej])/2],
                                             [Ex_field[0][Nei] + Ex_field[0][Nej], Ey_field[0][Nei] + Ey_field[0][Nej], Ez_field[0][Nei] + Ey_field[0][Nej]],
                                             [Bx_field[0][Nei] + Bx_field[0][Nej], By_field[0][Nei] + By_field[0][Nej], Bz_field[0][Nei]] + By_field[0][Nej],
                                             dt)
                ax, ay, az = electron_particle.acceleration()
                vx, vy, vz = electron_particle.velocity([ax, ay, az])
                xe, ye, ze = electron_particle.position([((vx + ve_x[Nej])/2), ((vy + ve_y[Nej])/2), ((vz + ve_z[Nej])/2)])

                ve_avg_x = (vx - ve_x[Nei])
                ve_avg_y = (vy - ve_y[Nei])
                ve_avg_z = (vz - ve_z[Nei])

                x_avg = (xe - x_electron[Nei])
                y_avg = (ye - y_electron[Nei])
                z_avg = (ze - z_electron[Nei])

                ve_x[Nei] = ve_avg_x
                ve_y[Nei] = ve_avg_y
                ve_z[Nei] = ve_avg_z

                x_electron[Nei] = x_avg
                y_electron[Nei] = y_avg
                z_electron[Nei] = z_avg

            elif Nei == Nej:
                break
    print(ve_x)
    #for the proton - proton interaction
    for Npi in range(num_particles[1]):

        x1 = (x_proton[Npi] + ((l1 - 2*a)/(b * Nx)))
        y1 = (y_proton[Npi] + ((2*b)/Ny))
        z1 = (z_proton[Npi] + ((2*b)/Ny))

        #Electric Field between Electron and Electrode
        Sa_proton_field = Field([0, 0, a], [x1, y1, z1])
        Sb_proton_field = Field([0, 0, l1 - a], [x1, y1, z1])

        Ex1, Ey1, Ez1 = Sa_proton_field.E(Q)
        Ex2, Ey2, Ez2 = Sb_proton_field.E(-Q)

        Ex_field[1][Npi] = (Ex1 + Ex2)
        Ey_field[1][Npi] = (Ey1 + Ey2)
        Ez_field[1][Npi] = (Ez1 + Ez2)

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

        Bx_field[1][Npi] = (0.5*(B1_x + B2_x))
        By_field[1][Npi] = (0.5*(B1_y + B2_y))
        Bz_field[1][Npi] = (0.5*(B1_z + B2_z))

        Uqpx1, Uqpy1, Uqpz1 = Sa_proton_field.U(Q, e)
        Uqpx2, Uqpy2, Uqpz2 = Sb_proton_field.U(-Q, e)

        Uqp_x[Npi] = (Uqpx1 + Uqpx2)
        Uqp_y[Npi] = (Uqpy1 + Uqpy2)
        Uqp_z[Npi] = (Uqpz1 + Uqpz2)
        U_qp[Npi] = np.sqrt(np.power((Uqpx1 + Uqpx2),2) + np.power((Uqpy1 + Uqpy2),2) + np.power((Uqpz1 + Uqpz2),2))

        vp_x[Npi] = (0.5*(np.sqrt((2*np.abs(Uqpx1))/mp) - np.sqrt((2*np.abs(Uqpx2))/mp)))
        vp_y[Npi] = (0.5*(np.sqrt((2*np.abs(Uqpy1))/mp) - np.sqrt((2*np.abs(Uqpy2))/mp)))
        vp_z[Npi] = (0.5*(np.sqrt((2*np.abs(Uqpz1))/mp) - np.sqrt((2*np.abs(Uqpz2))/mp)))

        for Npj in range(num_particles[0]):

            x2 = (x_proton[Npj] + ((l1 - 2*a)/(b * Nx)))
            y2 = (y_proton[Npj] + ((2*b)/Ny))
            z2 = (z_proton[Npj] + ((2*b)/Ny))

            if not(Npi == Npj):
                pp_interaction = Field([x1, y1, z1], [x2, y2, z2])
                Uppx, Uppy, Uppz = pp_interaction.U(e, e)

                Upp_x[Npi][Npj] = Uppx
                Upp_y[Npi][Npj] = Uppy
                Upp_z[Npi][Npj] = Uppz
                U_pp[Npi][Npj] = np.sqrt(np.power(Uppx,2) + np.power(Uppy,2) + np.power(Uppz,2))

                proton_particle = Particle(mp, e, [(vp_x[Npi] + vp_x[Npj])/2, (vp_y[Npi] + vp_y[Npj])/2, (vp_z[Npi] + vp_z[Npj])/2],
                                           [Ex_field[1][Npi] + Ex_field[1][Npj], Ey_field[1][Npi] + Ey_field[1][Npj], Ez_field[1][Npi] + Ez_field[1][Npj]],
                                           [Bx_field[1][Npi] + Bx_field[1][Npj], By_field[1][Npi] + By_field[1][Npj], Bz_field[1][Npi] + Bz_field[1][Npi]],
                                           dt)

                ax, ay, az = proton_particle.acceleration()
                vx, vy, vz = proton_particle.velocity([ax, ay, az])
                xp, yp, zp = proton_particle.position([((vx + vp_x[Npj])/2), ((vy + vp_y[Npj])/2), ((vz + vp_z[Npj])/2)])

                vp_avg_x = (vx - vp_x[Npi])
                vp_avg_y = (vy - vp_y[Npi])
                vp_avg_z = (vz - vp_z[Npi])

                x_avg = (xp - x_proton[Npi])
                y_avg = (yp - y_proton[Npi])
                z_avg = (zp - z_proton[Npi])

                vp_x[Npi] = vp_avg_x
                vp_y[Npi] = vp_avg_y
                vp_z[Npi] = vp_avg_z

                x_proton[Npi] = x_avg
                y_proton[Npi] = y_avg
                z_proton[Npi] = z_avg

            elif Npi == Npj:
                break

    #for the proton - electron interaction
    for Ne in range(num_particles[0]):

        xe = (x_electron[Ne] + ((l1 - 2*a)/(b * Nx)))
        ye = (y_electron[Ne] + ((2*b)/Ny))
        ze = (z_electron[Ne] + ((2*b)/Ny))

        #Electric Field between Electron and Electrode
        Sa_electron_field = Field([0, 0, a], [xe, ye, ze])
        Sb_electron_field = Field([0, 0, l1 - a], [xe, ye, ze])

        Exe1, Eye1, Eze1 = Sa_electron_field.E(Q)
        Exe2, Eye2, Eze2 = Sb_electron_field.E(-Q)

        Ex_field[0][Ne] = (Exe1 + Exe2)
        Ey_field[0][Ne] = (Eye1 + Eye2)
        Ez_field[0][Ne] = (Eze1 + Eze2)

        Bxe1_1, Bye1_1, Bze1_1 = Sa_electron_field.B1(I)
        Bxe2_1, Bye2_1, Bze2_1 = Sb_electron_field.B1(I)

        Bxe1_2, Bye1_2, Bze1_2 = Sa_electron_field.B2(I)
        Bxe2_2, Bye2_2, Bze2_2 = Sb_electron_field.B2(I)

        Bxe1 = Bxe1_1 + Bxe2_1
        Bye1 = Bye1_1 + Bye2_1
        Bze1 = Bze1_1 + Bze2_1

        Bxe2 = Bxe1_2 - Bxe2_2
        Bye2 = Bye1_2 - Bye2_2
        Bze2 = Bze1_2 - Bze2_2

        Bx_field[0][Ne] = (0.5*(Bxe1 + Bxe2))
        By_field[0][Ne] = (0.5*(Bye1 + Bye2))
        Bz_field[0][Ne] = (0.5*(Bze1 + Bze2))

        Uqex1, Uqey1, Uqez1 = Sa_electron_field.U(Q, -e)
        Uqex2, Uqey2, Uqez2 = Sb_electron_field.U(-Q, -e)

        Uqe_x[Ne] = (Uqex1 + Uqex1)
        Uqe_y[Ne] = (Uqey1 + Uqey1)
        Uqe_z[Ne] = (Uqez1 + Uqez1)
        U_qe[Ne] = np.sqrt(np.power((Uqex2 + Uqex1),2) + np.power((Uqey2 + Uqey1),2) + np.power((Uqez2 + Uqez1),2))

        ve_x[Nei] = (0.5*(np.sqrt((2*np.abs(Uqex2))/me) - np.sqrt((2*np.abs(Uqex1))/me)))
        ve_y[Nei] = (0.5*(np.sqrt((2*np.abs(Uqey2))/me) - np.sqrt((2*np.abs(Uqey1))/me)))
        ve_z[Nei] = (0.5*(np.sqrt((2*np.abs(Uqez2))/me) - np.sqrt((2*np.abs(Uqez1))/me)))

        for Np in range(num_particles[1]):

            xp = (x_proton[Np] + ((l1 - 2*a)/(b * Nx)))
            yp = (y_proton[Np] + ((2*b)/Ny))
            zp = (z_proton[Np] + ((2*b)/Ny))

            Sa_proton_field = Field([0, 0, a], [xp, yp, zp])
            Sb_proton_field = Field([0, 0, l1 - a], [xp, yp, zp])

            # vector function of an electric field
            Exp1, Eyp1, Ezp1 = Sa_proton_field.E(Q)
            Exp2, Eyp2, Ezp2 = Sb_proton_field.E(-Q)

            # x,y,z of an electric field
            Exp = Exp1 + Exp2
            Eyp = Eyp1 + Eyp2
            Ezp = Ezp1 + Ezp2

            Ex_field[1][Np] = Exp
            Ey_field[1][Np] = Eyp
            Ez_field[1][Np] = Ezp

            #calculate vector function of magnetic field perpendicular to the electric field
            Bxp1_1, Byp1_1, Bzp1_1 = Sa_proton_field.B1(I)
            Bxp2_1, Byp2_1, Bzp2_1 = Sb_proton_field.B1(I)

            Bxp1 = Bxp1_1 + Bxp2_1
            Byp1 = Byp1_1 + Byp2_1
            Bzp1 = Bzp1_1 + Bzp2_1

            #calculate vector function of magnetic field of the electromagnet
            Bxp1_2, Byp1_2, Bzp1_2 = Sa_proton_field.B2(I)
            Bxp2_2, Byp2_2, Bzp2_2 = Sb_proton_field.B2(I)

            Bxp2 = Bxp1_2 - Bxp2_2
            Byp2 = Byp1_2 - Byp2_2
            Bzp2 = Bzp1_2 - Bzp2_2

            Bxp = Bxp1 + Bxp2
            Byp = Byp1 + Byp2
            Bzp = Bzp1 + Bzp2

            Bx_field[1][Np] = Bxp
            By_field[1][Np] = Byp
            Bz_field[1][Np] = Bzp

            Uqpx1, Uqpy1, Uqpz1 = Sa_proton_field.U(Q, e)
            Uqpx2, Uqpy2, Uqpz2 = Sb_proton_field.U(-Q, e)

            Uqp_x[Np] = (Uqpx1 + Uqpx2)
            Uqp_y[Np] = (Uqpy1 + Uqpy2)
            Uqp_z[Np] = (Uqpz1 + Uqpz2)
            U_qp[Np] = np.sqrt(np.power((Uqpx1 + Uqpx2),2) + np.power((Uqpy1 + Uqpy2),2) + np.power((Uqpz1 + Uqpz2),2))

            vp_x[Npi] = (0.5*(np.sqrt((2*np.abs(Uqpx1))/mp) - np.sqrt((2*np.abs(Uqpx2))/mp)))
            vp_y[Npi] = (0.5*(np.sqrt((2*np.abs(Uqpy1))/mp) - np.sqrt((2*np.abs(Uqpy2))/mp)))
            vp_z[Npi] = (0.5*(np.sqrt((2*np.abs(Uqpz1))/mp) - np.sqrt((2*np.abs(Uqpz2))/mp)))

            #Electric Potential Energy between proton and electron
            pe_interaction = Field([xp, yp, zp], [xe, ye, ze])
            Upex, Upey, Upez = pe_interaction.U(-e, e)

            Upe_x[Ne][Np] = Upex
            Upe_y[Ne][Np] = Upey
            Upe_z[Ne][Np] = Upez
            U_pe[Ne][Np] = np.sqrt(np.power(Upex,2) + np.power(Upey,2) + np.power(Upez,2))

            electron_particle = Particle(me, -e, [(ve_x[Ne] + vp_x[Np])/2, (ve_y[Ne] + vp_y[Np])/2, (ve_z[Ne] + vp_z[Np])/2],
                                         [Ex_field[0][Ne] + Ex_field[1][Np], Ey_field[0][Ne] + Ey_field[1][Np], Ez_field[0][Ne] + Ez_field[1][Np]],
                                         [Bx_field[0][Ne] + Bx_field[1][Np], By_field[0][Ne] + By_field[1][Np], Bz_field[0][Ne] + Bz_field[1][Np]],
                                         dt)

            proton_particle = Particle(mp, e, [(ve_x[Ne] + vp_x[Np])/2, (ve_y[Ne] + vp_y[Np])/2, (ve_z[Ne] + vp_z[Np])/2],
                                       [Ex_field[0][Ne] + Ex_field[1][Np], Ey_field[0][Ne] + Ey_field[1][Np], Ez_field[0][Ne] + Ez_field[1][Np]],
                                         [Bx_field[0][Ne] + Bx_field[1][Np], By_field[0][Ne] + By_field[1][Np], Bz_field[0][Ne] + Bz_field[1][Np]],
                                       dt)

            ae_x, ae_y, ae_z = electron_particle.acceleration()
            ap_x, ap_y, ap_z = proton_particle.acceleration()

            vex, vey, vez = electron_particle.velocity([ae_x, ae_y, ae_z])
            vpx, vpy, vpz = proton_particle.velocity([ap_x, ap_y, ap_z])

            xe, ye, ze = electron_particle.position([(vex + vpx)/2, (vey + vpy)/2, (vez + vpz)/2])
            xp, yp, zp = proton_particle.position([((vex + vpx)/2), ((vey + vpy)/2), ((vez + vpz)/2)])

            ve_avg_x = (vex - ve_x[Ne])
            ve_avg_y = (vey - ve_y[Ne])
            ve_avg_z = (vez - ve_z[Ne])

            vp_avg_x = (vpx - vp_x[Np])
            vp_avg_y = (vpy - vp_y[Np])
            vp_avg_z = (vpz - vp_z[Np])

            xe_avg = (xe - x_electron[Ne])
            ye_avg = (ye - y_electron[Ne])
            ze_avg = (ze - z_electron[Ne])

            xp_avg = (xp - x_proton[Np])
            yp_avg = (yp - y_proton[Np])
            zp_avg = (zp - z_proton[Np])

            ve_x[Ne] = ve_avg_x
            ve_y[Ne] = ve_avg_y
            ve_z[Ne] = ve_avg_z

            vp_x[Np] = vp_avg_x
            vp_y[Np] = vp_avg_y
            vp_z[Np] = vp_avg_z

            x_electron[Ne] = xe_avg
            y_electron[Ne] = ye_avg
            z_electron[Ne] = ze_avg

            x_proton[Np] = xp_avg
            y_proton[Np] = yp_avg
            z_proton[Np] = yp_avg

    #-----------------------------Particle - Particle Interaction -----------------------------
    #----------------------------------------END--------------------------------------------
    
    #-----------------------------Relativistic Particle Field System----------------------
    
    #Relativistic Langrangian
    
    
    
    
    #----------------------------------------END------------------------------------------

        
    # Periodic boundary conditions
    x_electron = np.mod(x_electron, x_electron + 2*b) - b
    y_electron = np.mod(y_electron, y_electron + 2*b) - b
    z_electron = np.mod(z_electron - a, l1 - 2*a) + a

    x_proton = np.mod(x_proton, x_proton + 2*b) - b
    y_proton = np.mod(y_proton, y_proton + 2*b) - b
    z_proton = np.mod(z_proton - a, l1 - 2*a) + a
    
    # Plot particle positions
    axis.scatter(z_electron, y_electron, x_electron, s = 10, marker = 'o', color = "blue")
    axis.scatter(z_proton, y_proton, x_proton, s = 10, marker = 'o', color = "red")
    
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
