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

# x,y,z proton initial position
x_proton = np.random.uniform(-b, b, num_particles[1])
y_proton = np.random.uniform(-b, b, num_particles[1])
z_proton = np.random.uniform(a, l1 - a, num_particles[1])

#----------------------------------------------------End Particle System------------------------------------------------

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
    start_time = time.time()
    ax.cla()
    
    #-----------------------------Particle - Field Interaction -----------------------------
    # Calculate electric field due to electron's charges and positions
    for Nei in range(num_particles[0]):
        
        x = (x_electron[Nei]/(b * Nx))
        y = (y_electron[Nei]/(b * Ny))
        z = (z_electron[Nei]/(l1 * Nz))
        
        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):

            # vector function of an electric field
            Ex1, Ey1, Ez1 = E(Q, [0, 0, a], x, y, z)
            Ex2, Ey2, Ez2 = E(-Q, [0, 0, l1 -a], x, y, z)

            # x,y,z of an electric field
            Ex_field[0][grid_x] = (Ex1 + Ex2)
            Ey_field[0][grid_y] = (Ey1 + Ey2)
            Ez_field[0][grid_z] = (Ez1 + Ez2)

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

            Bx_field[0][grid_x] = (Bx1 + Bx2)
            By_field[0][grid_y] = (By1 + By2)
            Bz_field[0][grid_z] = (Bz1 + Bz2)

            Uqe_x[Ne] = (Uqex1 - Uqex2)
            Uqe_y[Ne] = (Uqey1 - Uqey2)
            Uqe_z[Ne] = (Uqez1 - Uqez2)
            U_qe[Ne] = np.sqrt(np.power((Uqex1 - Uqex2),2) + np.power((Uqey1 - Uqey2),2) + np.power((Uqez1 - Uqez2),2))
            
            ve_x[Ne] = (np.sqrt((2*np.abs(Uqex1))/me) - np.sqrt((2*np.abs(Uqex2))/me))
            ve_y[Ne] = (np.sqrt((2*np.abs(Uqey1))/me) - np.sqrt((2*np.abs(Uqey2))/me))
            ve_z[Ne] = (np.sqrt((2*np.abs(Uqez1))/me) - np.sqrt((2*np.abs(Uqez2))/me))
                    
    
    # Update particle positions and velocities
    for Nei in range(num_particles[0]):
        
        x = (x_electron[Nei]/(b * Nx))
        y = (y_electron[Nei]/(b * Ny))
        z = (z_electron[Nei]/(l1 * Nz))

        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)

        if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):
            xe, ye, ze, vx_e, vy_e, vz_e = electron(
                                       [vx_electron[Nei], vy_electron[Nei], vz_electron[Nei]],
                                       [Ex_field[0][grid_x], Ey_field[0][grid_y], Ez_field[0][grid_z]],
                                       [Bx_field[0][grid_x], By_field[0][grid_x], Bz_field[0][grid_y]]
                                        )
            
            #velocity of the particle in electrostatic force between electrode surface area and charged particle
            ve_x[Nei] += vx_e
            ve_y[Nei] += vy_e
            ve_z[Nei] += vz_e
            
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
        
        if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):

            # vector function of an electric field
            Ex1, Ey1, Ez1 = E(Q, [0, 0, a], x, y, z)
            Ex2, Ey2, Ez2 = E(-Q, [0, 0, l1 -a], x, y, z)

            Ex_field[1][grid_x] = (Ex1 + Ex2)
            Ey_field[1][grid_y] = (Ey1 + Ey2)
            Ez_field[1][grid_z] = (Ez1 + Ez2)

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

            Bx_field[1][grid_x] = (Bx1 + Bx2)
            By_field[1][grid_x] = (By1 + By2)
            Bz_field[1][grid_y] = (Bz1 + Bz2)
            
    # Update proton positions and velocities
    for Npi in range(num_particles[1]):
        
        x = (x_proton[Npi]/(b * Nx))
        y = (y_proton[Npi]/(b * Ny))
        z = (z_proton[Npi]/(l1 * Nz))

        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):
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

        if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):

            # vector function of an electric field
            Exie1, Eyie1, Ezie1 = E(Q, [0, 0, a], xie, yie, zie)
            Exie2, Eyie2, Ezie2 = E(-Q, [0, 0, l1 -a], xie, yie, zie)

            # x,y,z of an electric field
            Ex_field[0][grid_xie] = (Exie1 + Exie2)
            Ey_field[0][grid_yie] = (Eyie1 + Eyie2)
            Ez_field[0][grid_zie] = (Ezie1 + Ezie2)

            #calculate vector function of magnetic field perpendicular to the electric field
            Bxie1_1, Byie1_1, Bzie1_1 = B1(I, [0, 0, a], xie, yie, zie)
            Bxie2_1, Byie2_1, Bzie2_1 = B1(I, [0, 0, l1 -a], xie, yie, zie)
            Bxie1 = Bxie1_1 + Bxie2_1
            Byie1 = Byie1_1 + Byie2_1
            Bzie1 = Bzie1_1 + Bzie2_1

            #calculate vector function of magnetic field of the electromagnet
            Bxie1_2, Byie1_2, Bzie1_2 = B2(I, [0, 0, a], xie, yie, zie)
            Bxie2_2, Byie2_2, Bzie2_2 = B2(I, [0, 0, l1 -a], xie, yie, zie)
            Bxie2 = Bxie1_2 - Bxie2_2
            Byie2 = Byie1_2 - Byie2_2
            Bzie2 = Bzie1_2 - Bzie2_2

            Bx_field[0][grid_xie] = (Bxie1 + Bxie2)
            By_field[0][grid_xie] = (Byie1 + Byie2)
            Bz_field[0][grid_yie] = (Bzie1 + Bzie2)
            
            for Nej in range(num_particles[0]):
                xje = x_electron[Nej]/(b * Nx)
                yje = y_electron[Nej]/(b * Ny)
                zje = z_electron[Nej]/(l1 * Nz)

                grid_xje = int(np.abs(xje))
                grid_yje = int(np.abs(yje))
                grid_zje = int(zje)

                if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):

                    # vector function of an electric field
                    Exje1, Eyje1, Ezje1 = E(Q, [0, 0, a], xje, yje, zje)
                    Exje2, Eyje2, Ezje2 = E(-Q, [0, 0, l1 -a], xje, yje, zje)

                    # x,y,z of an electric field
                    Ex_field[0][grid_xje] = (Exje1 + Exje2)
                    Ey_field[0][grid_yje] = (Eyje1 + Eyje2)
                    Ez_field[0][grid_zje] = (Ezje1 + Ezje2)

                    #calculate vector function of magnetic field perpendicular to the electric field
                    Bxje1_1, Byje1_1, Bzje1_1 = B1(I, [0, 0, a], xje, yje, zje)
                    Bxje2_1, Byje2_1, Bzje2_1 = B1(I, [0, 0, l1 -a], xje, yje, zje)

                    Bxje1 = Bxje1_1 + Bxje2_1
                    Byje1 = Byje1_1 + Byje2_1
                    Bzje1 = Bzje1_1 + Bzje2_1

                    #calculate vector function of magnetic field of the electromagnet
                    Bxje1_2, Byje1_2, Bzje1_2 = B2(I, [0, 0, a], xje, yje, zje)
                    Bxje2_2, Byje2_2, Bzje2_2 = B2(I, [0, 0, l1 -a], xje, yje, zje)

                    Bxje2 = Bxje1_2 - Bxje2_2
                    Byje2 = Byje1_2 - Byje2_2
                    Bzje2 = Bzje1_2 - Bzje2_2

                    Bx_field[0][grid_xje] = (Bxje1 + Bxje2)
                    By_field[0][grid_xje] = (Byje1 + Byje2)
                    Bz_field[0][grid_yje] = (Bzje1 + Bzje2)
                    
                    #Electric Potential Energy between electron and grid
                    Ueex, Ueey, Ueez = U(-e, -e, [xie, yie, zie], xje, yje, zje)

                    if not(Nei == Nej):

                        #Electric Potential Energy between electron particles
                        Uee_x[Nei][Nej] = Ueex
                        Uee_y[Nei][Nej] = Ueey
                        Uee_z[Nei][Nej] = Ueez
                        U_ee[Nei][Nej] = (np.sqrt(np.power(Ueex, 2) + np.power(Ueey, 2) + np.power(Ueez, 2))/e)

                        xe1, ye1, ze1, vx_e1, vy_e1, vz_e1 = electron(
                                       [vx_electron[Nei], vy_electron[Nei], vz_electron[Nei]],
                                       [Ex_field[0][grid_xie], Ey_field[0][grid_yie], Ez_field[0][grid_zie]],
                                       [Bx_field[0][grid_xie], By_field[0][grid_xie], Bz_field[0][grid_yie]]
                                        )
                        xe2, ye2, ze2, vx_e2, vy_e2, vz_e2 = electron(
                                       [vx_electron[Nej], vy_electron[Nej], vz_electron[Nej]],
                                       [Ex_field[0][grid_xje], Ey_field[0][grid_yje], Ez_field[0][grid_zje]],
                                       [Bx_field[0][grid_xje], By_field[0][grid_xje], Bz_field[0][grid_yje]]
                                        )

                        #velocity of the particle in electrostatic force between electrode surface area and charged particle
                        vx_electron[Nei] += ((vx_e1 + vx_e2)/2)
                        vy_electron[Nei] += ((vy_e1 + vy_e2)/2)
                        vz_electron[Nei] += ((vz_e1 + vz_e2)/2)

                        #final position of the particle
                        x_electron[Nei] += ((xe1 + xe2)/2)
                        y_electron[Nei] += ((ye1 + ye2)/2)
                        z_electron[Nei] += ((ze1 + ze2)/2)

    #for the proton - proton interaction
    for Npi in range(num_particles[1]):
        
        xi_p = x_proton[Npi]/(b * Nx)
        yi_p = y_proton[Npi]/(b * Ny)
        zi_p = z_proton[Npi]/(l1 * Nz)

        grid_xip = int(np.abs(xi_p))
        grid_yip = int(np.abs(yi_p))
        grid_zip = int(zi_p)

        if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):

            # vector function of an electric field
            Exip1, Eyip1, Ezip1 = E(Q, [0, 0, a], xi_p, yi_p, zi_p)
            Exip2, Eyip2, Ezip2 = E(-Q, [0, 0, l1 -a], xi_p, yi_p, zi_p)

            # x,y,z of an electric field
            Ex_field[1][grid_xip] = (Exip1 + Exip2)
            Ey_field[1][grid_yip] = (Eyip1 + Eyip2)
            Ez_field[1][grid_zip] = (Ezip1 + Ezip2)

            #calculate vector function of magnetic field perpendicular to the electric field
            Bxip1_1, Byip1_1, Bzip1_1 = B1(I, [0, 0, a], xi_p, yi_p, zi_p)
            Bxip2_1, Byip2_1, Bzip2_1 = B1(I, [0, 0, l1 -a], xi_p, yi_p, zi_p)
            Bxip1 = Bxip1_1 + Bxip2_1
            Byip1 = Byip1_1 + Byip2_1
            Bzip1 = Bzip1_1 + Bzip2_1

            #calculate vector function of magnetic field of the electromagnet
            Bxip1_2, Byip1_2, Bzip1_2 = B2(I, [0, 0, a], xi_p, yi_p, zi_p)
            Bxip2_2, Byip2_2, Bzip2_2 = B2(I, [0, 0, l1 -a], xi_p, yi_p, zi_p)
            Bxip2 = Bxip1_2 - Bxip2_2
            Byip2 = Byip1_2 - Byip2_2
            Bzip2 = Bzip1_2 - Bzip2_2

            Bx_field[1][grid_xip] = (Bxip1 + Bxip2)
            By_field[1][grid_xip] = (Byip1 + Byip2)
            Bz_field[1][grid_yip] = (Bzip1 + Bzip2)
            
            for Npj in range(num_particles[0]):
                xjp = x_proton[Npj]/(b * Nx)
                yjp = y_proton[Npj]/(b * Ny)
                zjp = z_proton[Npj]/(l1 * Nz)

                grid_xjp = int(np.abs(xjp))
                grid_yjp = int(np.abs(yjp))
                grid_zjp = int(zjp)

                if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):

                    # vector function of an electric field
                    Exjp1, Eyjp1, Ezjp1 = E(Q, [0, 0, a], xjp, yjp, zjp)
                    Exjp2, Eyjp2, Ezjp2 = E(-Q, [0, 0, l1 -a], xjp, yjp, zjp)

                    # x,y,z of an electric field
                    Ex_field[1][grid_xjp] = (Exjp1 + Exjp2)
                    Ey_field[1][grid_yjp] = (Eyjp1 + Eyjp2)
                    Ez_field[1][grid_zjp] = (Ezjp1 + Ezjp2)

                    #calculate vector function of magnetic field perpendicular to the electric field
                    Bxjp1_1, Byjp1_1, Bzjp1_1 = B1(I, [0, 0, a], xjp, yjp, zjp)
                    Bxjp2_1, Byjp2_1, Bzjp2_1 = B1(I, [0, 0, l1 -a], xjp, yjp, zjp)
                    Bxjp1 = Bxjp1_1 + Bxjp2_1
                    Byjp1 = Byjp1_1 + Byjp2_1
                    Bzjp1 = Bzjp1_1 + Bzjp2_1

                    #calculate vector function of magnetic field of the electromagnet
                    Bxjp1_2, Byjp1_2, Bzjp1_2 = B2(I, [0, 0, a], xjp, yjp, zjp)
                    Bxjp2_2, Byjp2_2, Bzjp2_2 = B2(I, [0, 0, l1 -a], xjp, yjp, zjp)
                    Bxjp2 = Bxjp1_2 - Bxjp2_2
                    Byjp2 = Byjp1_2 - Byjp2_2
                    Bzjp2 = Bzjp1_2 - Bzjp2_2

                    Bx_field[1][grid_xjp] = (Bxjp1 + Bxjp2)
                    By_field[1][grid_xjp] = (Byjp1 + Byjp2)
                    Bz_field[1][grid_yjp] = (Bzjp1 + Bzjp2)
            
                    #Electric Potential between proton and grid
                    Uppx, Uppy, Uppz = U(e, e, [xi_p, yi_p, zi_p], xjp, yjp, zjp)

                    if not(Npi == Npj):

                        #Electric Potential Energy between proton particles
                        Upp_x[Npi][Npj] = Uppx
                        Upp_y[Npi][Npj] = Uppy
                        Upp_z[Npi][Npj] = Uppz
                        U_pp[Npi][Npj] = (np.sqrt(np.power(Uppx, 2) + np.power(Uppy, 2) + np.power(Uppz, 2))/e)
                        
                        xp1, yp1, zp1, vx_p1, vy_p1, vz_p1 = proton(
                                       [vx_proton[Npi], vy_proton[Npi], vz_proton[Npi]],
                                       [Ex_field[1][grid_xip], Ey_field[1][grid_yip], Ez_field[1][grid_zip]],
                                       [Bx_field[1][grid_xip], By_field[1][grid_xip], Bz_field[1][grid_yip]]
                                        )
                        xp2, yp2, zp2, vx_p2, vy_p2, vz_p2 = proton(
                                       [vx_proton[Npj], vy_proton[Npj], vz_proton[Npj]],
                                       [Ex_field[1][grid_xjp], Ey_field[1][grid_yjp], Ez_field[1][grid_zjp]],
                                       [Bx_field[1][grid_xjp], By_field[1][grid_xjp], Bz_field[1][grid_yjp]]
                                        )

                        #velocity of the particle in electrostatic force between electrode surface area and charged particle
                        vx_proton[Npi] += ((vx_p1 + vx_p2)/2)
                        vy_proton[Npi] += ((vy_p1 + vy_p2)/2)
                        vz_proton[Npi] += ((vz_p1 + vz_p2)/2)

                        #final position of the particle
                        x_proton[Npi] += ((xp1 + xp2)/2)
                        y_proton[Npi] += ((yp1 + yp2)/2)
                        z_proton[Npi] += ((zp1 + zp2)/2)

    #for the proton - electron interaction
    for Ne in range(num_particles[0]):
        xe = x_electron[Ne]/(b * Nx)
        ye = y_electron[Ne]/(b * Ny)
        ze = z_electron[Ne]/(l1 * Nz)

        grid_xe = int(np.abs(xe))
        grid_ye = int(np.abs(ye))
        grid_ze = int(ze)

        if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):

            # vector function of an electric field
            Exe1, Eye1, Eze1 = E(Q, [0, 0, a], xe, ye, ze)
            Exe2, Eye2, Eze2 = E(-Q, [0, 0, l1 -a], xe, ye, ze)

            # x,y,z of an electric field
            Ex_field[0][grid_xe] = (Exe1 + Exe2)
            Ey_field[0][grid_ye] = (Eye1 + Eye2)
            Ez_field[0][grid_ze] = (Eze1 + Eze2)

            #calculate vector function of magnetic field perpendicular to the electric field
            Bxe1_1, Bye1_1, Bze1_1 = B1(I, [0, 0, a], xe, ye, ze)
            Bxe2_1, Bye2_1, Bze2_1 = B1(I, [0, 0, l1 -a], xe, ye, ze)
            Bxe1 = Bxe1_1 + Bxe2_1
            Bye1 = Bye1_1 + Bye2_1
            Bze1 = Bze1_1 + Bze2_1

            #calculate vector function of magnetic field of the electromagnet
            Bxe1_2, Bye1_2, Bze1_2 = B2(I, [0, 0, a], xe, ye, ze)
            Bxe2_2, Bye2_2, Bze2_2 = B2(I, [0, 0, l1 -a], xe, ye, ze)
            Bxe2 = Bxe1_2 - Bxe2_2
            Bye2 = Bye1_2 - Bye2_2
            Bze2 = Bze1_2 - Bze2_2

            Bx_field[0][grid_xe] = (Bxe1 + Bxe2)
            By_field[0][grid_xe] = (Bye1 + Bye2)
            Bz_field[0][grid_ye] = (Bze1 + Bze2)

            for Np in range(num_particles[1]):
                xp = x_proton[Np]/(b * Nx)
                yp = y_proton[Np]/(b * Ny)
                zp = z_proton[Np]/(l1 * Nz)

                grid_xp = int(np.abs(xp))
                grid_yp = int(np.abs(yp))
                grid_zp = int(zp)

                if ((0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 <= grid_z < Nz)):

                    # vector function of an electric field
                    Exp1, Eyp1, Ezp1 = E(Q, [0, 0, a], xp, yp, zp)
                    Exp2, Eyp2, Ezp2 = E(-Q, [0, 0, l1 -a], xp, yp, zp)

                    # x,y,z of an electric field
                    Ex_field[1][grid_xp] = (Exp1 + Exp2)
                    Ey_field[1][grid_yp] = (Eyp1 + Eyp2)
                    Ez_field[1][grid_zp] = (Ezp1 + Ezp2)

                    #calculate vector function of magnetic field perpendicular to the electric field
                    Bxp1_1, Byp1_1, Bzp1_1 = B1(I, [0, 0, a], xp, yp, zp)
                    Bxp2_1, Byp2_1, Bzp2_1 = B1(I, [0, 0, l1 -a], xp, yp, zp)
                    Bxp1 = Bxp1_1 + Bxp2_1
                    Byp1 = Byp1_1 + Byp2_1
                    Bzp1 = Bzp1_1 + Bzp2_1

                    #calculate vector function of magnetic field of the electromagnet
                    Bxp1_2, Byp1_2, Bzp1_2 = B2(I, [0, 0, a], xp, yp, zp)
                    Bxp2_2, Byp2_2, Bzp2_2 = B2(I, [0, 0, l1 -a], xp, yp, zp)
                    Bxp2 = Bxp1_2 - Bxp2_2
                    Byp2 = Byp1_2 - Byp2_2
                    Bzp2 = Bzp1_2 - Bzp2_2

                    Bx_field[1][grid_xip] = (Bxip1 + Bxip2)
                    By_field[1][grid_xip] = (Byip1 + Byip2)
                    Bz_field[1][grid_yip] = (Bzip1 + Bzip2)
                    
                    Upex, Upey, Upez = U(e, -e, [-xe, -ye, -ze], xp, yp, zp)

                    #Electric Potential Energy between proton and electron
                    Upe_x[Ne][Np] = Upex
                    Upe_y[Ne][Np] = Upey
                    Upe_z[Ne][Np] = Upez
                    U_pe[Ne][Np] = (np.sqrt(np.power(Upex, 2) + np.power(Upey, 2) + np.power(Upez, 2))/e)

                    xe, ye, ze, vx_e, vy_e, vz_e = electron([vx_electron[Ne], vy_electron[Ne], vz_electron[Ne]],
                                       [Ex_field[0][grid_xe], Ey_field[0][grid_ye], Ez_field[0][grid_ze]],
                                       [Bx_field[0][grid_xe], By_field[0][grid_xe], Bz_field[0][grid_ye]]
                                        )

                    xp, yp, zp, vx_p, vy_p, vz_p = proton([vx_proton[Np], vy_proton[Np], vz_proton[Np]],
                                       [Ex_field[1][grid_xp], Ey_field[1][grid_yp], Ez_field[1][grid_zp]],
                                       [Bx_field[1][grid_xp], By_field[1][grid_xp], Bz_field[1][grid_yp]]
                                        )
                    #velocity of the particle in electrostatic force between electrode surface area and charged particle
                    vx_electron[Ne] += ((vx_e + vx_p)/2)
                    vy_electron[Ne] += ((vy_e + vy_p)/2)
                    vz_electron[Ne] += ((vz_e + vz_p)/2)

                    #final position of the particle
                    x_electron[Ne] += ((xe + xp)/2)
                    y_electron[Ne] += ((ye + yp)/2)
                    z_electron[Ne] += ((ze + zp)/2)

                    #velocity of the particle in electrostatic force between electrode surface area and charged particle
                    vx_proton[Np] += ((vx_e + vx_p)/2)
                    vy_proton[Np] += ((vy_e + vy_p)/2)
                    vz_proton[Np] += ((vz_e + vz_p)/2)

                    #final position of the particle
                    x_proton[Np] += ((xe + xp)/2)
                    y_proton[Np] += ((ye + yp)/2)
                    z_proton[Np] += ((ze + zp)/2)

    #-----------------------------Particle - Particle Interaction -----------------------------
    #----------------------------------------END--------------------------------------------
    
    #-----------------------------Relativistic Particle Field System----------------------
    
    #Relativistic Langrangian
    
    
    
    
    #----------------------------------------END------------------------------------------

    #-----------------------------------------START PRINTING THE FINAL RESULT------------------------------
    #print(f"Electron's Position({x_electron}, {y_electron}, {z_electron})")
    #print(f"Electron's Velocity({vx_electron}, {vy_electron}, {vz_electron})")
    #print(f"Proton's Position({x_proton}, {y_proton}, {z_proton})")
    #print(f"Proton's Velocity({vx_proton}, {vy_proton}, {vz_proton})")
    #print()
    #print(f"Electric Field at x-axis: {Ex_field[0,:]}")
    print(f"Electric Field at x-axis: {Ex_field[1,:]}")
    print()
    #print()
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
