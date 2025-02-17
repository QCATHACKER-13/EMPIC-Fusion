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
from matplotlib.animation import FuncAnimation, writers

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

#max pressure 760 mmHg delta_pressure = 25 mmHg 1 mmHg = 133.322 Pascal
# Number of grid points
Nx = 10000000
Ny = 10000000
Nz = 10000000

# parameters for the particles
num_particles = [10, 5]

# Electric field
Ex_field = np.zeros(Nx)
Ey_field = np.zeros(Ny)
Ez_field = np.zeros(Nz)

# Magnetic field
Bx_field = np.zeros(Nx)
By_field = np.zeros(Ny)
Bz_field = np.zeros(Nz)

B1x_field = np.zeros(Nx)
B1y_field = np.zeros(Ny)
B1z_field = np.zeros(Nz)

B2x_field = np.zeros(Nx)
B2y_field = np.zeros(Ny)
B2z_field = np.zeros(Nz)

#Electric potential of particle's interaction between them
Uee_x = [[0.0 for _ in range(num_particles[0])] for _ in range(num_particles[0])]
Uee_y = [[0.0 for _ in range(num_particles[0])] for _ in range(num_particles[0])]
Uee_z = [[0.0 for _ in range(num_particles[0])] for _ in range(num_particles[0])]

Upe_x = [[0.0 for _ in range(num_particles[0])] for _ in range(num_particles[1])]
Upe_y = [[0.0 for _ in range(num_particles[0])] for _ in range(num_particles[1])]
Upe_z = [[0.0 for _ in range(num_particles[0])] for _ in range(num_particles[1])]

Upp_x = [[0.0 for _ in range(num_particles[1])] for _ in range(num_particles[1])]
Upp_y = [[0.0 for _ in range(num_particles[1])] for _ in range(num_particles[1])]
Upp_z = [[0.0 for _ in range(num_particles[1])] for _ in range(num_particles[1])]

U_ee = [[0.0 for _ in range(num_particles[0])] for _ in range(num_particles[0])]
U_pe = [[0.0 for _ in range(num_particles[0])] for _ in range(num_particles[1])]
U_pp = [[0.0 for _ in range(num_particles[1])] for _ in range(num_particles[1])]

# x,y electron initial position
x_electron = np.random.uniform(-b, b, num_particles[0])
y_electron = np.random.uniform(-b, b, num_particles[0])
z_electron = np.random.uniform(a, l1 - a, num_particles[0])

# Initial x,y electron velocity
vx_electron = np.random.uniform(0, x_electron/dt, num_particles[0])
vy_electron = np.random.uniform(0, y_electron/dt, num_particles[0])
vz_electron = np.random.uniform(0, z_electron/dt, num_particles[0])

# x,y proton initial position
x_proton = np.random.uniform(-b, b, num_particles[1])
y_proton = np.random.uniform(-b, b, num_particles[1])
z_proton = np.random.uniform(a, l1 - a, num_particles[1])

# Initial x,y proton velocity
vx_proton = np.random.uniform(0, x_proton/dt, num_particles[1])
vy_proton = np.random.uniform(0, y_proton/dt, num_particles[1])
vz_proton = np.random.uniform(0, z_proton/dt, num_particles[1])

print(f"Electron's Position({x_electron}, {y_electron}, {z_electron})")
print(f"Electron's Velocity({vx_electron}, {vy_electron}, {vz_electron})")
print(f"Proton's Position({x_proton}, {y_proton}, {z_proton})")
print(f"Proton's Velocity({vx_proton}, {vy_proton}, {vz_proton})")

# Create the figure and axes
fig = plt.figure(dpi = 100)
ax = fig.add_subplot(projection='3d')

#----------------------------------------------------Start Field System------------------------------------------------
#Electric Field
def E(q, a1, X, Y, Z):
    r = np.sqrt(np.power(X-a1[0],2) + np.power(Y - a1[1],2) + np.power(Z - a1[2],2))
    E_x = 2*k0*q*(X-a1[0])/np.power(r,3)
    E_y = k0*q*(Y-a1[1])/np.power(r,3)
    E_z = k0*q*(Z-a1[2])/np.power(r,3)
    return E_x,E_y,E_z

#Electric Potential Energy for Energy Particle Matrix Identity
def U(q1, q2, a2, X, Y, Z):
    r = np.sqrt(np.power(X-a2[0],2) + np.power(Y - a2[1],2) + np.power(Z - a2[2],2))
    U_x = (k0*q1*q2*(X - a2[0]))/np.power(r,3)
    U_y = (k0*q1*q2*(Y - a2[1]))/np.power(r,3)
    U_z = (k0*q1*q2*(Z - a2[2]))/np.power(r,3)
    return U_x, U_y, U_z

def B1(I0, a3, X, Y, Z):
    r = np.sqrt(np.power(X-a3[0],2) + np.power(Y - a3[1],2) + np.power(Z - a3[2],2))
    B_x1 = (mu_0*I0*(Y - a3[1]))/(2*np.pi*np.power(r,3))
    B_y1 = -(mu_0*I0*(X - a3[0]))/(2*np.pi*np.power(r,3))
    B_z1 = np.zeros_like(r)
    return B_x1,B_y1,B_z1

#Magnetic Vector Field of the External Magnetic Field
def B2(I0, a4, X, Y, Z):
    r = np.sqrt(4*np.power(X-a4[0],2) + 4*np.power(Y - a4[1],2) + np.power(Z - a4[2],2))
    B_x2 = (mu_0*I0*N*(X - a4[0]))/(2*np.pi*np.power(r,3))
    B_y2 = (mu_0*I0*N*(Y - a4[1]))/(2*np.pi*np.power(r,3))
    B_z2 = (mu_0*I0*N*(Z - a4[2]))/(2*np.pi*np.power(r,3))
    return B_x2,B_y2,B_z2

#Magnetic Vector Potential of the Induced Magnetic Field
def A1(I0, a4, X, Y, Z):
    r = np.sqrt(np.power(X-a4[0],2) + np.power(Y - a4[1],2) + np.power(Z - a4[2],2))
    A1_x = -(mu_0*I0*(X - a4[0])*(Z - a4[2]))/(2*np.pi*r*(np.power(X - a4[0],2) + np.power(Y - a4[1],2)))
    A1_y = -(mu_0*I0*(Y - a4[1])*(Z - a4[2]))/(2*np.pi*r*(np.power(X - a4[0],2) + np.power(Y - a4[1],2)))
    A1_z = ((mu_0*I0)/(2*np.pi*r))*((np.power(Y-a4[1],2)/(np.power(X - a4[0],2) + np.power(Z - a4[2],2))) + (np.power(X - a4[0],2)/(np.power(Y - a4[1],2) + np.power(Z - a4[2],2))))
    return A1_x, A1_y, A1_z

#Magnetic Vector Potential of the External Magnetic Field
def A2(I0, a5, X, Y, Z):
    r = np.sqrt(np.power(X-a5[0],2) + np.power(Y - a5[1],2) + np.power(Z - a5[2],2))
    A2_x = ((mu_0*I0*N)/(2*np.pi*r))*((((Y - a5[1])*(Z - a5[2]))/(np.power(X-a5[0],2) + np.power(Y - a5[1],2))) - (((Y - a5[1])*(Z - a5[2]))/(np.power(X - a5[0],2) + np.power(Z - a5[2],2))))
    A2_y = ((mu_0*I0*N)/(2*np.pi*r))*((((X - a5[0])*(Z - a5[2]))/(np.power(Y-a5[1],2) + np.power(Z - a5[2],2))) - (((X - a5[0])*(Z - a5[2]))/(np.power(X - a5[0],2) + np.power(Y - a5[1],2))))
    A2_z = ((mu_0*I0*N)/(2*np.pi*r))*((((X - a5[0])*(Y - a5[1]))/(np.power(X - a5[0],2) + np.power(Z - a5[2],2))) + (((X - a5[0])*(Y - a5[1]))/(np.power(Y - a5[1],2) + np.power(Z - a5[2],2))))
    return A2_x, A2_y, A2_z

#Electric Potential
def V(q, a6, X, Y, Z):
    r = np.sqrt(np.power(X - a6[0],2) + np.power(Y - a6[1],2) + np.power(Z - a6[2],2))
    V_x = 2*k0*q*(X-a6[0])/np.power(r,2)
    V_y = k0*q*(Y-a6[1])/np.power(r,2)
    V_z = k0*q*(Z-a6[2])/np.power(r,2)
    return V_x, V_y,V_z
#----------------------------------------------------End Field System-----------------------------------------------------

#----------------------------------------------------Start Particle System------------------------------------------------
#Relativistic Lagrangian Particle Field System
def electron(v0_e, E_e, B_e):
    #resultant of initial velocity
    v0_e_resultant = np.sgrt(np.power(v0_e[0],2) + np.power(v0_e[1],2) + np.power(v0_e[2],2))
    
    #acceleration of the particle in electrostatic force between electrode surface area and charged particle
    ax_e = (-e/me)*(E_e[0] + (0.5*((v0_e[1]*B_e[2]) - (v0_e[2]*B_e[1]))))
    ay_e = (-e/me)*(E_e[1] + (0.5*((v0_e[2]*B_e[0]) - (v0_e[0]*B_e[2]))))
    az_e = (-e/me)*(E_e[2] + (0.5*((v0_e[0]*B_e[1]) - (v0_e[1]*B_e[0]))))

    #velocity of the particle in electrostatic force between electrode surface area and charged particle
    vx_e = (0.5*ax_e*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
    vy_e = (0.5*ay_e*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
    vz_e = (0.5*az_e*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))

    #final position of the particle
    x_e = (vx_e*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
    y_e = (vy_e*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
    z_e = (vz_e*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
    
    return x_e, y_e, z_e, vx_e, vy_e, vz_e

def proton(v0_p, E_p, B_p):
    #resultant of initial velocity
    v0_p_resultant = np.sgrt(np.power(v0_p[0],2) + np.power(v0_p[1],2) + np.power(v0_p[2],2))
    
    #acceleration of the particle in electrostatic force between electrode surface area and charged particle
    ax_p = (e/me)*(E_p[0] + (0.5*((v0_p[1]*B_p[2]) - (v0_p[2]*B_p[1]))))
    ay_p = (e/me)*(E_p[1] + (0.5*((v0_p[2]*B_p[0]) - (v0_p[0]*B_p[2]))))
    az_p = (e/me)*(E_p[2] + (0.5*((v0_p[0]*B_p[1]) - (v0_p[1]*B_p[0]))))

    #velocity of the particle in electrostatic force between electrode surface area and charged particle
    vx_p = (0.5*ax_p*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
    vy_p = (0.5*ay_p*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
    vz_p = (0.5*az_p*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))

    #final position of the particle
    x_p = (vx_p*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
    y_p = (vy_p*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
    z_p = (vz_p*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
    
    return x_p, y_p, z_p, vx_p, vy_p, vz_p 

def Lagragian(ue, up, Ae, Ap, Ve, Vp, U):
    #for U[0] is electron - electron interaction, U[1] is proton - electron interaction and U[2] is proton - proton interaction
    V_electron = np.sqrt(np.power(Ve[0],2) + np.power(Ve[1],2) + np.power(Ve[2],2))
    V_proton = np.sqrt(np.power(Vp[0],2) + np.power(Vp[1],2) + np.power(Vp[2],2))
    K_electron = (0.5*me*np.power(c0, 2)) -(e*V_electron) + e*(ue[0]*Ae[0] + ue[1]*Ae[1] + ue[2]*Ae[2])
    K_proton = (0.5*mp*np.power(c0, 2)) + (e*V_proton) - e*(up[0]*Ap[0] + up[1]*Ap[1] + up[2]*Ap[2])
    L_electron = K_electron - (U[0] + U[1] + U[2])
    L_proton = K_proton - (U[0] + U[1] + U[2])
    L = K_electron + K_proton - U[0] - U[1] - U[2]
    return L, L_electron, L_proton

#Schrodinger's Equation in Relativistic Hamilthonian System
def Hamilthonian(qe, qp, ke, kp, Ve, Vp, U):
    #for U[0] is electron - electron interaction, U[1] is proton - electron interaction and U[2] is proton - proton interaction
    V_electron = np.sqrt(np.power(Ve[0],2) + np.power(Ve[1],2) + np.power(Ve[2],2))
    V_proton = np.sqrt(np.power(Vp[0],2) + np.power(Vp[1],2) + np.power(Vp[2],2))
    K_electron = (me*np.power(c0,2)) - ((np.power(h_bar,2)/(2*me))*(np.power(ke[0],2) + np.power(ke[1],2) + np.power(ke[2],2))) - e*V_electron
    K_proton = (mp*np.power(c0,2)) - ((np.power(h_bar,2)/(2*mp))*(np.power(kp[0],2) + np.power(kp[1],2) + np.power(kp[2],2))) + e*V_proton
    H_electron = K_electron + U[0] + U[1] + U[2]
    H_proton = K_proton + U[0] + U[1] + U[2]
    H = K_electron + K_proton + U[0] + U[1] + U[2]
    return H, H_electron, H_proton
#----------------------------------------------------End Particle System------------------------------------------------

# Main simulation loop
for step in range(steps):
    ax.cla()

    #--------------------Start Field System-------------------
    #Initialization
    # Clear electric field
    Ex_field[:][:] = 0.0
    Ey_field[:][:] = 0.0
    Ez_field[:][:] = 0.0

    # Clear magnetic field
    Bx_field[:][:] = 0.0
    By_field[:][:] = 0.0
    Bz_field[:][:] = 0.0
    
    B1x_field[:][:] = 0.0
    B1y_field[:][:] = 0.0
    B1z_field[:][:] = 0.0
    
    B2x_field[:][:] = 0.0
    B2y_field[:][:] = 0.0
    B2z_field[:][:] = 0.0
    #--------------------End Field System-------------------

    #initialize the postion, velocity and electric potential as particle's property
    #--------------------------Start Particle's Property---------------------------
    
    #---------------------------End Particle's Property----------------------------

    #time to process
    start_time = time.time()
    #-----------------------------Particle - Field Interaction -----------------------------
    # Calculate electric field due to electron's charges and positions
    for Nei in range(num_particles[0]):
        
        x = (x_electron[Nei]/(b * Nx))
        y = (y_electron[Nei]/(b * Ny))
        z = (z_electron[Nei]/(l1 * Nz))
        
        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 < grid_z <= Nz):

            # vector function of an electric field
            Ex1, Ey1, Ez1 = E(Q, [0, 0, a], x, y, z)
            Ex2, Ey2, Ez2 = E(-Q, [0, 0, l1 -a], x, y, z)

            # x,y,z of an electric field
            Ex_field[0][grid_x] = Ex1 + Ex2
            Ey_field[0][grid_y] = Ey1 + Ey2
            Ez_field[0][grid_z] = Ez1 + Ez2

            #calculate vector function of magnetic field perpendicular to the electric field
            Bx1_1, By1_1, Bz1_1 = B1(I, [0, 0, a], x, y, z)
            Bx2_1, By2_1, Bz2_1 = B1(I, [0, 0, l1 -a], x, y, z)
            Bx1 = Bx1_1 + Bx2_1
            By1 = By1_1 + By2_1
            Bz1 = Bz1_1 + Bz2_1

            #calculate vector function of magnetic field of the electromagnet
            Bx1_2, By1_2, Bz1_2 = B2(I, [0, 0, a], x, y, z)
            Bx2_2, By2_2, Bz2_2 = B2(I, [0, 0, l1 -a], x, y, z)
            Bx2 = Bx1_2 - Bx2_2
            By2 = By1_2 - By2_2
            Bz2 = Bz1_2 - Bz2_2

            #calculate the B1 addition of B2
            B1x_field[0][grid_x] = Bx1
            B1y_field[0][grid_x] = By1
            B1z_field[0][grid_y] = Bz1

            B2x_field[0][grid_x] = Bx2
            B2y_field[0][grid_x] = By2
            B2z_field[0][grid_y] = Bz2

            Bx_field[0][grid_x] = (Bx1 + Bx2)
            By_field[0][grid_x] = (By1 + By2)
            Bz_field[0][grid_y] = (Bz1 + Bz2)
                    
    
    # Update particle positions and velocities
    for Nei in range(num_particles[0]):
        
        x = (x_electron[Nei]/(b * Nx))
        y = (y_electron[Nei]/(b * Ny))
        z = (z_electron[Nei]/(l1 * Nz))

        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 < grid_z <= Nz):
            xe, ye, ze, vx_e, vy_e, vz_e = electron(
                                       [vx_electron[Nei], vy_electron[Nei], vz_electron[Nei]],
                                       [Ex_field[0][grid_x], Ey_field[0][grid_y], Ez_field[0][grid_z]],
                                       [Bx_field[0][grid_x], By_field[0][grid_x], Bz_field[0][grid_y]]
                                        )
            
            #velocity of the particle in electrostatic force between electrode surface area and charged particle
            vx_electron[Nei] += vx_e
            vy_electron[Nei] += vy_e
            vz_electron[Nei] += vz_e

            #final position of the particle
            x_electron[Nei] += xe
            y_electron[Nei] += ye
            z_electron[Nei] += ze

    print(f"Electron's Position({x_electron}, {y_electron}, {z_electron})")
    print(f"Electron's Velocity({vx_electron}, {vy_electron}, {vz_electron})")
    print(f"Proton's Position({x_proton}, {y_proton}, {z_proton})")
    print(f"Proton's Velocity({vx_proton}, {vy_proton}, {vz_proton})")
    
    # Calculate electric field due to proton's charges and positions
    for Npi in range(num_particles[1]):
        
        x = (x_proton[Npi]/(b * Nx))
        y = (y_proton[Npi]/(b * Ny))
        z = (z_proton[Npi]/(l1 * Nz))
        
        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 < grid_z <= Nz):

            # vector function of an electric field
            Ex1, Ey1, Ez1 = E(Q, [0, 0, a], x, y, z)
            Ex2, Ey2, Ez2 = E(-Q, [0, 0, l1 -a], x, y, z)

            # x,y,z of an electric field
            Ex_field[1][grid_x] += Ex1 + Ex2
            Ey_field[1][grid_y] += Ey1 + Ey2
            Ez_field[1][grid_z] += Ez1 + Ez2

            #calculate vector function of magnetic field perpendicular to the electric field
            Bx1_1, By1_1, Bz1_1 = B1(I, [0, 0, a], x, y, z)
            Bx2_1, By2_1, Bz2_1 = B1(I, [0, 0, l1 -a], x, y, z)
            Bx1 = Bx1_1 + Bx2_1
            By1 = By1_1 + By2_1
            Bz1 = Bz1_1 + Bz2_1

            #calculate vector function of magnetic field of the electromagnet
            Bx1_2, By1_2, Bz1_2 = B2(I, [0, 0, a], x, y, z)
            Bx2_2, By2_2, Bz2_2 = B2(I, [0, 0, l1 - a], x, y, z)
            Bx2 = Bx1_2 - Bx2_2
            By2 = By1_2 - By2_2
            Bz2 = Bz1_2 - Bz2_2

            #calculate the B1 addition of B2
            B1x_field[1][grid_x] += Bx1
            B1y_field[1][grid_x] += By1
            B1z_field[1][grid_y] += Bz1

            B2x_field[1][grid_x] += Bx2
            B2y_field[1][grid_x] += By2
            B2z_field[1][grid_y] += Bz2

            Bx_field[1][grid_x] += (Bx1 + Bx2)
            By_field[1][grid_x] += (By1 + By2)
            Bz_field[1][grid_y] += (Bz1 + Bz2)
            
    # Update proton positions and velocities
    for Npi in range(num_particles[1]):
        
        x = (x_proton[Npi]/(b * Nx))
        y = (y_proton[Npi]/(b * Ny))
        z = (z_proton[Npi]/(l1 * Nz))

        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 < grid_z <= Nz):
            xp, yp, zp, vx_p, vy_p, vz_p = proton(
                                       [vx_proton[Npi], vy_proton[Npi], vz_proton[Npi]],
                                       [Ex_field[1][grid_x], Ey_field[1][grid_y], Ez_field[1][grid_z]],
                                       [Bx_field[1][grid_x], By_field[1][grid_x], Bz_field[1][grid_y]]
                                        )

            #velocity of the particle in electrostatic force between electrode surface area and charged particle
            vx_proton[Npi] += vx_p
            vy_proton[Npi] += vy_p
            vz_proton[Npi] += vz_p

            #final position of the particle
            x_proton[Npi] += xp
            y_proton[Npi] += yp
            z_proton[Npi] += zp
            
    #-----------------------------Particle - Field Interaction -----------------------------
    #----------------------------------------END--------------------------------------------
    
    #-----------------------------Particle - Particle Interaction -----------------------------
    #for electron - electron interaction
    for Nei in range(num_particles[0]):
        
        xie = x_electron[Nei]/(b * Nx)
        yie = y_electron[Nei]/(b * Ny)
        zie = z_electron[Nei]/(l1 * Nz)

        grid_xie = int(np.abs(xie))
        grid_yie = int(np.abs(yie))
        grid_zie = int(zie)

        if (0 <= grid_xie < Nx) and (0 <= grid_yie < Ny) and (0 < grid_zie <= Nz):

            #Electric Potential Energy between electron and grid
            U1ex1, U1ey1, U1ez1 = U(Q, -e, [0, 0, a], xie, yie, zie)
            U1ex2, U1ey2, U1ez2 = U(-Q, -e, [0, 0, l1 - a], xie, yie, zie)
            
            for Nej in range(num_particles[0]):
                xje = x_electron[Nej]/(b * Nx)
                yje = y_electron[Nej]/(b * Ny)
                zje = z_electron[Nej]/(l1 * Nz)

                grid_xje = int(np.abs(xje))
                grid_yje = int(np.abs(yje))
                grid_zje = int(zje)

                if (0 <= grid_xje < Nx) and (0 <= grid_yje < Ny) and (0 < grid_zje <= Nz):

                    #Electric Potential Energy between electron and grid
                    U2ex1, U2ey1, U2ez1 = U(Q, -e, [0, 0, a], xje, yje, zje)
                    U2ex2, U2ey2, U2ez2 = U(-Q, -e, [0, 0, l1 - a], xje, yje, zje)

                    Ua_xe = (U1ex1 - U1ex2)
                    Ua_ye = (U1ey1 - U1ey2)
                    Ua_ze = (U1ez1 - U1ez2)

                    Ub_xe = (U2ex1 - U2ex2)
                    Ub_ye = (U2ey1 - U2ey2)
                    Ub_ze = (U2ez1 - U2ez2)

                    if not(Nei == Nej):

                        #Electric Potential Energy between electron particles
                        Uee_x[Nei][Nej] += ((Ua_xe - Ub_xe)*(-e/Q))
                        Uee_y[Nei][Nej] += ((Ua_ye - Ub_ye)*(-e/Q))
                        Uee_z[Nei][Nej] += ((Ua_ze - Ub_ze)*(-e/Q))
                        U_ee[Nei][Nej] = np.sqrt(np.power(Uee_x[Nei][Nej], 2) + np.power(Uee_y[Nei][Nej], 2) + np.power(Uee_x[Nei][Nej], 2))/e

                        xe1, ye1, ze1, vx_e1, vy_e1, vz_e1 = electron(
                                       [vx_electron[Nei], vy_electron[Nei], vz_electron[Nei]],
                                       [Ex_field[0][grid_xie], Ey_field[0][grid_yie], Ez_field[0][grid_zie]],
                                       [Bx_field[0][grid_xie], By_field[0][grid_xie], Bz_field[0][grid_yie]]
                                        )
                        xe2_final, ye2_final, ze2_final, vx_e2_final, vy_e2_final, vz_e2_final = electron(
                                       [vx_electron[Nej], vy_electron[Nej], vz_electron[Nej]],
                                       [Ex_field[0][grid_xje], Ey_field[0][grid_yje], Ez_field[0][grid_zje]],
                                       [Bx_field[0][grid_xje], By_field[0][grid_xje], Bz_field[0][grid_yje]]
                                        )

                        #velocity of the particle in electrostatic force between electrode surface area and charged particle
                        vx_electron[Nei] += (vx_e1_final + vx_e2_final)
                        vy_electron[Nei] += (vy_e1_final + vy_e2_final)
                        vz_electron[Nei] += (vz_e1_final + vz_e2_final)

                        #final position of the particle
                        x_electron[Nei] += (xe1_final + xe2_final)
                        y_electron[Nei] += (ye1_final + ye2_final)
                        z_electron[Nei] += (ze1_final + ze2_final)

                    elif Nei == Nej:
                        break

    #for the proton - proton interaction
    for Npi in range(num_particles[1]):
        
        xi_p = x_proton[Npi]/(b * Nx)
        yi_p = y_proton[Npi]/(b * Ny)
        zi_p = z_proton[Npi]/(l1 * Nz)

        grid_xip = int(np.abs(xi_p))
        grid_yip = int(np.abs(yi_p))
        grid_zip = int(zi_p)

        #Electric Potential between proton and grid
        U1px1, U1py1, U1pz1 = U(Q, e, [0, 0, a], xi_p, yi_p, zi_p)
        U1px2, U1py2, U1pz2 = U(-Q, e, [0, 0, l1 - a], xi_p, yi_p, zi_p)

        if (0 <= grid_xip < Nx) and (0 <= grid_yip < Ny) and (0 < grid_zip <= Nz):
            
            for Npj in range(num_particles[0]):
                xjp = x_proton[Npj]/(b * Nx)
                yjp = y_proton[Npj]/(b * Ny)
                zjp = z_proton[Npj]/(l1 * Nz)

                grid_xjp = int(np.abs(xjp))
                grid_yjp = int(np.abs(yjp))
                grid_zjp = int(zjp)

                if (0 <= grid_xjp < Nx) and (0 <= grid_yjp < Ny) and (0 < grid_zjp <= Nz):

                    #Electric Potential between proton and grid
                    U2px1, U2py1, U2pz1 = U(Q, e, [0, 0, a], xjp, yjp, zjp)
                    U2px2, U2py2, U2pz2 = U(-Q, e, [0, 0, l1 - a], xjp, yjp, zjp)
                    
                    Ua_xp = (U1px1 - U1px2)
                    Ua_yp = (U1py1 - U1py2)
                    Ua_zp = (U1pz1 - U1pz2)

                    Ub_xp = (U2px1 - U2px2)
                    Ub_yp = (U2py1 - U2py2)
                    Ub_zp = (U2pz1 - U2pz2)

                    if not(Npi == Npj):

                        #Electric Potential Energy between proton particles
                        Upp_x[Npi][Npj] = ((Ua_xp - Ub_xp)*(e/Q))
                        Upp_y[Npi][Npj] = ((Ua_yp - Ub_yp)*(e/Q))
                        Upp_z[Npi][Npj] = ((Ua_zp - Ub_zp)*(e/Q))
                        U_pp[Npi][Npj] = np.sqrt(np.power(Upp_x[Npi][Npj], 2) + np.power(Upp_y[Npi][Npj], 2) + np.power(Upp_x[Npi][Npj], 2))/e
                        
                        xp1_final, yp1_final, zp1_final, vx_p1_final, vy_p1_final, vz_p1_final = proton(
                                       [vx_proton[Npi], vy_proton[Npi], vz_proton[Npi]],
                                       [Ex_field[1][grid_xip], Ey_field[1][grid_yip], Ez_field[1][grid_zip]],
                                       [Bx_field[1][grid_xip], By_field[1][grid_xip], Bz_field[1][grid_yip]]
                                        )
                        xp2_final, yp2_final, zp2_final, vx_p2_final, vy_p2_final, vz_p2_final = proton(
                                       [vx_proton[Npj], vy_proton[Npj], vz_proton[Npj]],
                                       [Ex_field[1][grid_xjp], Ey_field[1][grid_yjp], Ez_field[1][grid_zjp]],
                                       [Bx_field[1][grid_xjp], By_field[1][grid_xjp], Bz_field[1][grid_yjp]]
                                        )

                        #velocity of the particle in electrostatic force between electrode surface area and charged particle
                        vx_proton[Npi] += (vx_p1_final + vx_p2_final)
                        vy_proton[Npi] += (vy_p1_final + vy_p2_final)
                        vz_proton[Npi] += (vz_p1_final + vz_p2_final)

                        #final position of the particle
                        x_proton[Npi] += (xp1_final + xp2_final)
                        y_proton[Npi] += (yp1_final + yp2_final)
                        z_proton[Npi] += (zp1_final + zp2_final)

                    elif Npi == Npj:
                        break

    #for the proton - electron interaction
    for Ne in range(num_particles[0]):

        xe = x_electron[Ne]/(b * Nx)
        ye = y_electron[Ne]/(b * Ny)
        ze = z_electron[Ne]/(l1 * Nz)

        grid_xe = int(np.abs(xe))
        grid_ye = int(np.abs(ye))
        grid_ze = int(ze)
        

        if (0 <= grid_xe < Nx) and (0 <= grid_ye < Ny) and (0 < grid_ze <= Nz):

            #Electric Potential between electron and grid
            Uex1, Uey1, Uez1 = U(Q, -e, [0, 0, a], xe, ye, ze)
            Uex2, Uey2, Uez2 = U(-Q, -e, [0, 0, l1 - a], xe, ye, ze)
            
            for Np in range(num_particles[1]):

                xp = x_proton[Np]/(b * Nx)
                yp = y_proton[Np]/(b * Ny)
                zp = z_proton[Np]/(l1 * Nz)

                grid_xp = int(np.abs(xp))
                grid_yp = int(np.abs(yp))
                grid_zp = int(zp)

                if (0 <= grid_xp < Nx) and (0 <= grid_yp < Ny) and (0 < grid_zp <= Nz):

                    #Electric Potential Energy between proton and grid
                    Upx1, Upy1, Upz1 = U(Q, e, [0, 0, a], xp, yp, zp)
                    Upx2, Upy2, Upz2 = U(-Q, e, [0, 0, l1 - a], xp, yp, zp)

                    U_xp = (Upx1 - Upx2)
                    U_yp = (Upy1 - Upy2)
                    U_zp = (Upz1 - Upz2)

                    U_xe = (Uex1 - Uex2)
                    U_ye = (Uey1 - Uey2)
                    U_ze = (Uez1 - Uez2)

                    #Electric Potential Energy between proton and electron
                    Upe_x[Ne][Np] = ((U_xp - U_xe)*(e/Q))
                    Upe_y[Ne][Np] = ((U_yp - U_ye)*(e/Q))
                    Upe_z[Ne][Np] = ((U_zp - U_ze)*(e/Q))
                    U_pe[Ne][Np] = np.sqrt(np.power(Upe_x[Ne][Np], 2) + np.power(Upe_y[Ne][Np], 2) + np.power(Upe_z[Ne][Np], 2))/e

                    xe_final, ye_final, ze_final, vx_e_final, vy_e_final, vz_e_final = electron(
                                       [vx_electron[Ne], vy_electron[Ne], vz_electron[Ne]],
                                       [Ex_field[0][grid_xe], Ey_field[0][grid_ye], Ez_field[0][grid_ze]],
                                       [Bx_field[0][grid_xe], By_field[0][grid_xe], Bz_field[0][grid_ye]]
                                        )

                    xp_final, yp_final, zp_final, vx_p_final, vy_p_final, vz_p_final = proton(
                                       [vx_proton[Np], vy_proton[Np], vz_proton[Np]],
                                       [Ex_field[1][grid_xp], Ey_field[1][grid_yp], Ez_field[1][grid_zp]],
                                       [Bx_field[1][grid_xp], By_field[1][grid_xp], Bz_field[1][grid_yp]]
                                        )
                    #velocity of the particle in electrostatic force between electrode surface area and charged particle
                    vx_electron[Ne] += (vx_e_final + vx_p_final)
                    vy_electron[Ne] += (vy_e_final + vy_p_final)
                    vz_electron[Ne] += (vz_e_final + vz_p_final)

                    #final position of the particle
                    x_electron[Ne] += (xe_final + xp_final)
                    y_electron[Ne] += (ye_final + yp_final)
                    z_electron[Ne] += (ze_final + zp_final)

                    #velocity of the particle in electrostatic force between electrode surface area and charged particle
                    vx_proton[Np] += (vx_e_final + vx_p_final)
                    vy_proton[Np] += (vy_e_final + vy_p_final)
                    vz_proton[Np] += (vz_e_final + vz_p_final)

                    #final position of the particle
                    x_proton[Np] += (xe_final + xp_final)
                    y_proton[Np] += (ye_final + yp_final)
                    z_proton[Np] += (ze_final + zp_final)

    #-----------------------------Particle - Particle Interaction -----------------------------
    #----------------------------------------END--------------------------------------------
    
    #-----------------------------Relativistic Particle Field System----------------------
    
    #Relativistic Langrangian
    
    
    
    
    #----------------------------------------END------------------------------------------

    #-----------------------------------------START PRINTING THE FINAL RESULT------------------------------
    print(f"Electron's Position({x_electron}, {y_electron}, {z_electron})")
    print(f"Electron's Velocity({vx_electron}, {vy_electron}, {vz_electron})")
    print(f"Proton's Position({x_proton}, {y_proton}, {z_proton})")
    print(f"Proton's Velocity({vx_proton}, {vy_proton}, {vz_proton})")

    #for electron_electron_PE in U_ee:
        #print(f"Electron - Electron Electric Potential Energy is {electron_electron_PE/np.power(10,3)} KeV")
    #for electron_proton_PE in U_pe:
        #print(f"Electron - Proton Electric Potential Energy is {electron_proton_PE/np.power(10,3)} KeV")
    #for proton_proton_PE in U_pp:
        #print(f"Proton - Proton Electric Potential Energy is {proton_proton_PE/np.power(10,3)} KeV")
    #-----------------------------------------END PRINTING THE FINAL RESULT------------------------------
    
    # Periodic boundary conditions
    x_electron = np.mod(x_electron, x_electron + 2*b) - b
    y_electron = np.mod(y_electron, y_electron + 2*b) - b
    z_electron = np.mod(z_electron - a, l1 - 2*a) + a

    x_proton = np.mod(x_proton, x_proton + 2*b) - b
    y_proton = np.mod(y_proton, y_proton + 2*b) - b
    z_proton = np.mod(z_proton - a, l1 - 2*a) + a
    
    #X.append(x_particles)
    #Y.append(y_particles)
    #Z.append(z_particles) np.linspace(0, 1*np.power(10, 12))
    
    # Plot particle positions
    ax.scatter(z_electron, y_electron, x_electron, s = 10, marker = 'o', color = "blue")
    ax.scatter(z_proton, y_proton, x_proton, s = 10, marker = 'o', color = "red")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{step} of {steps} steps")
    print(f"The time finish: {execution_time}")
    print()
    
    ax.set_xlim(0, l1)
    ax.set_ylim(-b, b)
    ax.set_zlim(-b, b)
    ax.set_xlabel("z - Position")
    ax.set_ylabel("y - Position")
    ax.set_zlabel("x - Position")
    ax.set_title("Particle Motion with Interaction")
    plt.pause(0.001)

plt.show()
