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
dt = 1e-3# time step
steps = int(np.ceil(60/dt))

print("Simulation of Nuclear Fusion Reaction")
print(f"The simulation is begin with a total frames = {steps}")
print()

#max pressure 760 mmHg delta_pressure = 25 mmHg 1 mmHg = 133.322 Pascal
# Number of grid points
Nx = 1000
Ny = 1000
Nz = 1000

# parameters for the particles
num_particles = [10, 5]

#Initialization of the Field System
# Electric field
Ex_electron_field = np.zeros(Nx)
Ey_electron_field = np.zeros(Ny)
Ez_electron_field = np.zeros(Nz)

Ex_proton_field = np.zeros(Nx)
Ey_proton_field = np.zeros(Ny)
Ez_proton_field = np.zeros(Nz)

# Magnetic field
Bx_electron_field = np.zeros(Nx)
By_electron_field = np.zeros(Ny)
Bz_electron_field = np.zeros(Nz)

Bx_proton_field = np.zeros(Nx)
By_proton_field = np.zeros(Ny)
Bz_proton_field = np.zeros(Nz)

#Electric potential of particle's interaction between them
Uee_x = np.zeros(Nx)
Uee_y = np.zeros(Ny)
Uee_z = np.zeros(Nz)

Upe_x = np.zeros(Nx)
Upe_y = np.zeros(Ny)
Upe_z = np.zeros(Nz)

Upp_x = np.zeros(Nx)
Upp_y = np.zeros(Ny)
Upp_z = np.zeros(Nz)

U_ee = np.zeros(Nx)
U_pe = np.zeros(Ny)
U_pp = np.zeros(Nz)

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
    B_x2 = (mu_0*I0*n*(X - a4[0]))/(2*np.pi*np.power(r,3))
    B_y2 = (mu_0*I0*n*(Y - a4[1]))/(2*np.pi*np.power(r,3))
    B_z2 = (mu_0*I0*n*(Z - a4[2]))/(2*np.pi*np.power(r,3))
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
    v0_e_resultant = np.sqrt(np.power(v0_e[0],2) + np.power(v0_e[1],2) + np.power(v0_e[2],2))
    
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
    v0_p_resultant = np.sqrt(np.power(v0_p[0],2) + np.power(v0_p[1],2) + np.power(v0_p[2],2))
    
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

# x,y,z electron initial position
x_electron = np.random.uniform(-b, b, num_particles[0])
y_electron = np.random.uniform(-b, b, num_particles[0])
z_electron = np.random.uniform(a, l1 - a, num_particles[0])

# Initial x,y,z electron velocity
vx_electron = np.random.uniform(0, ve, num_particles[0])
vy_electron = np.random.uniform(0, ve, num_particles[0])
vz_electron = np.random.uniform(0, ve, num_particles[0])

# x,y,z proton initial position
x_proton = np.random.uniform(-b, b, num_particles[1])
y_proton = np.random.uniform(-b, b, num_particles[1])
z_proton = np.random.uniform(a, l1 - a, num_particles[1])

# Initial x,y,z proton velocity
vx_proton = np.random.uniform(0, vp, num_particles[1])
vy_proton = np.random.uniform(0, vp, num_particles[1])
vz_proton = np.random.uniform(0, vp, num_particles[1])
#----------------------------------------------------End Particle System------------------------------------------------

print(f"Electron's Position({x_electron}, {y_electron}, {z_electron})")
print(f"Electron's Velocity({vx_electron}, {vy_electron}, {vz_electron})")
print(f"Proton's Position({x_proton}, {y_proton}, {z_proton})")
print(f"Proton's Velocity({vx_proton}, {vy_proton}, {vz_proton})")

# Main simulation loop
for step in range(steps):
    
    # Clear electric field
    Ex_electron_field[:] = 0.0
    Ey_electron_field[:] = 0.0
    Ez_electron_field[:] = 0.0

    Ex_proton_field[:] = 0.0
    Ey_proton_field[:] = 0.0
    Ez_proton_field[:] = 0.0

    # Clear magnetic field
    Bx_electron_field[:] = 0.0
    By_electron_field[:] = 0.0
    Bz_electron_field[:] = 0.0

    Bx_proton_field[:] = 0.0
    By_proton_field[:] = 0.0
    Bz_proton_field[:] = 0.0
    
    #time to process
    start_time = time.time()
    ax.cla()
    
    #-----------------------------Particle - Field Interaction -----------------------------
    # Calculate electric field due to electron's charges and positions
    for Nei in range(num_particles[0]):
        
        x = (x_electron[Nei]/(b * Nx))
        y = (y_electron[Nei]/(b *Ny))
        z = (z_electron[Nei]/(l1*Nz))
        
        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 < grid_z <= Nz):

            # vector function of an electric field
            Ex1, Ey1, Ez1 = E(Q, [0, 0, a], x, y, z)
            Ex2, Ey2, Ez2 = E(-Q, [0, 0, l1 -a], x, y, z)

            # x,y,z of an electric field
            Ex = Ex1 + Ex2
            Ey = Ey1 + Ey2
            Ez = Ez1 + Ez2

            Ex_electron_field[grid_x] = Ex
            Ey_electron_field[grid_y] = Ey
            Ez_electron_field[grid_z] = Ez

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

            Bx = Bx1 + Bx2
            By = By1 + By2
            Bz = Bz1 + Bz2

            Bx_electron_field[grid_x] = Bx
            By_electron_field[grid_x] = By
            Bz_electron_field[grid_y] = Bz
                    
    
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
                                       [Ex_electron_field[grid_x], Ey_electron_field[grid_y], Ez_electron_field[grid_z]],
                                       [Bx_electron_field[grid_x], By_electron_field[grid_x], Bz_electron_field[grid_y]]
                                        )
            
            #velocity of the particle in electrostatic force between electrode surface area and charged particle
            vx_electron[Nei] += vx_e
            vy_electron[Nei] += vy_e
            vz_electron[Nei] += vz_e
            
            #final position of the particle
            x_electron[Nei] += xe
            y_electron[Nei] += ye
            z_electron[Nei] += ze
                

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
            Ex = Ex1 + Ex2
            Ey = Ey1 + Ey2
            Ez = Ez1 + Ez2

            Ex_proton_field[grid_x] = Ex
            Ey_proton_field[grid_y] = Ey
            Ez_proton_field[grid_z] = Ez

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

            Bx = Bx1 + Bx2
            By = By1 + By2
            Bz = Bz1 + Bz2

            Bx_proton_field[grid_x] = Bx
            By_proton_field[grid_x] = By
            Bz_proton_field[grid_y] = Bz
            
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
                                       [Ex_proton_field[grid_x], Ey_proton_field[grid_y], Ez_proton_field[grid_z]],
                                       [Bx_proton_field[grid_x], By_proton_field[grid_x], Bz_proton_field[grid_y]]
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
    
   

    #-----------------------------------------START PRINTING THE FINAL RESULT------------------------------
    print(f"Electron's Position({x_electron}, {y_electron}, {z_electron})")
    print(f"Electron's Velocity({vx_electron}, {vy_electron}, {vz_electron})")
    print(f"Proton's Position({x_proton}, {y_proton}, {z_proton})")
    print(f"Proton's Velocity({vx_proton}, {vy_proton}, {vz_proton})")
    print()
    print(f"Electric Field at x-axis: {Ex_electron_field}")
    print(f"Electric Field at x-axis: {Ex_proton_field}")
    print()
    print()
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
