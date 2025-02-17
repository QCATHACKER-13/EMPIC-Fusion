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

Estate_ee = np.zeros(Nx)
Estate_pe = np.zeros(Ny)
Estate_pp = np.zeros(Nz)

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

# Create the figure and axes
fig = plt.figure(dpi = 100)
ax = fig.add_subplot(projection='3d')

#Electric Field
def E(q, a1, X, Y, Z):
    r = np.sqrt(np.power(X-a1[0],2) + np.power(Y - a1[1],2) + np.power(Z - a1[2],2))
    E_x = 2*k0*q*(X-a1[0])/np.power(r,3)
    E_y = k0*q*(Y-a1[1])/np.power(r,3)
    E_z = k0*q*(Z-a1[2])/np.power(r,3)
    return E_x,E_y,E_z

#Electric Potential Energy for Energy Particle Matrix Identity
def U(q1, q2, a2, X, Y, Z):
    r = np.sqrt(np.power(X-a1[0],2) + np.power(Y - a1[1],2) + np.power(Z - a1[2],2))
    U_x = k0*q1*q2/np.power(r,3)
    U_y = k0*q1*q2/np.power(r,3)
    U_z = k0*q1*q2/np.power(r,3)
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
    V_x = 2*k0*q*(X-a1[0])/np.power(r,2)
    V_y = k0*q*(Y-a1[1])/np.power(r,2)
    V_z = k0*q*(Z-a1[2])/np.power(r,2)
    return V_x, V_y,V_z

def U(q1, q2, a7, X, Y, Z):
    r = np.sqrt(np.power(X - a7[0],2) + np.power(Y - a7[1],2) + np.power(Z - a7[2],2))
    U_x = k0*q1*q2*((X - a7[0])/np.power(r,2))
    U_y = k0*q1*q2*((Y - a7[1])/np.power(r,2))
    U_z = k0*q1*q2*((Z - a7[2])/np.power(r,2))
    return U_x, U_y, U_z

#Relativistic Lagrangian Particle Field System
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

# Main simulation loop
for step in range(steps):
    
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

    # Clear electric potential
    Uee_x[:][:] = 0.0
    Uee_y[:][:] = 0.0
    Uee_z[:][:] = 0.0

    Upe_x[:][:] = 0.0
    Upe_y[:][:] = 0.0
    Upe_z[:][:] = 0.0

    Upp_x[:][:] = 0.0
    Upp_y[:][:] = 0.0
    Upp_z[:][:] = 0.0

    U_ee[:][:] = 0.0
    U_pe[:][:] = 0.0
    U_pp[:][:] = 0.0

    #time to process
    start_time = time.time()
    #-----------------------------Particle - Field Interaction -----------------------------
    # Calculate electric field due to electron's charges and positions
    for Nei in range(num_particles[0]):
        
        x = x_electron[Nei]/(b * Nx)
        y = y_electron[Nei]/(b * Ny)
        z = z_electron[Nei]/(l1 * Nz)
        
        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 < grid_z <= Nz):

            # vector function of an electric field
            Ex1, Ey1, Ez1 = E(Q, [0, 0, a], x, y, z)
            Ex2, Ey2, Ez2 = E(-Q, [0, 0, l1 -a], x, y, z)

            # x,y,z of an electric field
            Ex_field[0][grid_x] += Ex1 + Ex2
            Ey_field[0][grid_y] += Ey1 + Ey2
            Ez_field[0][grid_z] += Ez1 + Ez2

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
            B1x_field[0][grid_x] += Bx1
            B1y_field[0][grid_x] += By1
            B1z_field[0][grid_y] += Bz1

            B2x_field[0][grid_x] += Bx2
            B2y_field[0][grid_x] += By2
            B2z_field[0][grid_y] += Bz2

            Bx_field[0][grid_x] += (Bx1 + Bx2)
            By_field[0][grid_x] += (By1 + By2)
            Bz_field[0][grid_y] += (Bz1 + Bz2)
                    
    
    # Update particle positions and velocities
    for Nei in range(num_particles[0]):

        ax.cla()
        
        x = x_electron[Nei]/(b * Nx)
        y = y_electron[Nei]/(b * Ny)
        z = z_electron[Nei]/(l1 * Nz)

        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 < grid_z <= Nz):
            #acceleration of the particle in electrostatic force between electrode surface area and charged particle
            x_acceleration = (-e/me)*(Ex_field[0][grid_x] + (0.5*((vy_electron[Nei]*Bz_field[0][grid_z]) - (vz_electron[Nei]*By_field[0][grid_y]))))
            y_acceleration = (-e/me)*(Ey_field[0][grid_y] + (0.5*((vz_electron[Nei]*Bx_field[0][grid_x]) - (vx_electron[Nei]*Bz_field[0][grid_z]))))
            z_acceleration = (-e/me)*(Ez_field[0][grid_z] + (0.5*((vx_electron[Nei]*By_field[0][grid_y]) - (vy_electron[Nei]*Bx_field[0][grid_x]))))

            #resultant of intial velocity of electron
            v0_e_resultant = np.sqrt(np.power(vx_electron[Nei],2) + np.power(vy_electron[Nei],2) + np.power(vz_electron[Nei],2))


            #velocity of the particle in electrostatic force between electrode surface area and charged particle
            vx_electron[Nei] += (0.5*x_acceleration*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
            vy_electron[Nei] += (0.5*y_acceleration*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
            vz_electron[Nei] += (0.5*z_acceleration*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))

            #final position of the particle
            x_electron[Nei] += (vx_electron[Nei]*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
            y_electron[Nei] += (vy_electron[Nei]*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
            z_electron[Nei] += (vz_electron[Nei]*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
                

    # Calculate electric field due to proton's charges and positions
    for Npi in range(num_particles[1]):
        
        x = x_proton[Npi]/(b * Nx)
        y = y_proton[Npi]/(b * Ny)
        z = z_proton[Npi]/(l1 * Nz)
        
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

        ax.cla()
        
        x = x_proton[Npi]/(b * Nx)
        y = y_proton[Npi]/(b * Ny)
        z = z_proton[Npi]/(l1 * Nz)

        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 < grid_z <= Nz):
            #acceleration of the particle in electrostatic force between electrode surface area and charged particle
            x_acceleration = (e/me)*(Ex_field[1][grid_x] + (0.5*((vy_electron[Npi]*Bz_field[1][grid_z]) - (vz_electron[Npi]*By_field[1][grid_y]))))
            y_acceleration = (e/me)*(Ey_field[1][grid_y] + (0.5*((vz_electron[Npi]*Bx_field[1][grid_x]) - (vx_electron[Npi]*Bz_field[1][grid_z]))))
            z_acceleration = (e/me)*(Ez_field[1][grid_z] + (0.5*((vx_electron[Npi]*By_field[1][grid_y]) - (vy_electron[Npi]*Bx_field[1][grid_x]))))

            #resultant of intial velocity of proton
            v0_p_resultant = np.sqrt(np.power(vx_electron[Nei],2) + np.power(vy_electron[Nei],2) + np.power(vz_electron[Nei],2))
            
            #velocity of the particle in electrostatic force between electrode surface area and charged particle
            vx_proton[Npi] += (0.5*x_acceleration*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
            vy_proton[Npi] += (0.5*y_acceleration*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
            vz_proton[Npi] += (0.5*z_acceleration*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))

            #final position of the particle
            x_proton[num_proton] += (vx_proton[Npi]*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
            y_proton[num_proton] += (vy_proton[Npi]*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
            z_proton[num_proton] += (vz_proton[Npi]*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
    #-----------------------------Particle - Field Interaction -----------------------------
    #----------------------------------------END--------------------------------------------
    
    #-----------------------------Particle - Particle Interaction -----------------------------
    #for electron - electron interaction
    for Nei in range(num_particles[0]):

        ax.cla()
        
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

                    Ua_xe = U1ex1 - U1ex2
                    Ua_ye = U1ey1 - U1ey2
                    Ua_ze = U1ez1 - U1ez2

                    Ub_xe = U2ex1 - U2ex2
                    Ub_ye = U2ey1 - U2ey2
                    Ub_ze = U2ez1 - U2ez2

                    #acceleration of the electron in electrostatic force between electron particles
                    xi_acceleration = (-e/me)*(Ex_field[0][grid_xie] + (0.5*((vy_electron[Nei]*Bz_field[0][grid_zie]) - (vz_electron[Nei]*By_field[0][grid_yie]))))
                    yi_acceleration = (-e/me)*(Ey_field[0][grid_yie] + (0.5*((vz_electron[Nei]*Bx_field[0][grid_xie]) - (vx_electron[Nei]*Bz_field[0][grid_zie]))))
                    zi_acceleration = (-e/me)*(Ez_field[0][grid_zie] + (0.5*((vx_electron[Nei]*By_field[0][grid_yie]) - (vy_electron[Nei]*Bx_field[0][grid_xie]))))

                    xj_acceleration = (-e/me)*(Ex_field[0][grid_xje] + (0.5*((vy_electron[Nej]*Bz_field[0][grid_zje]) - (vz_electron[Nej]*By_field[0][grid_yje]))))
                    yj_acceleration = (-e/me)*(Ey_field[0][grid_yje] + (0.5*((vz_electron[Nej]*Bx_field[0][grid_xje]) - (vx_electron[Nej]*Bz_field[0][grid_zje]))))
                    zj_acceleration = (-e/me)*(Ez_field[0][grid_zje] + (0.5*((vx_electron[Nej]*By_field[0][grid_yje]) - (vy_electron[Nej]*Bx_field[0][grid_xje]))))

                    if not(Nei == Nej):

                        #Electric Potential Energy between electron particles
                        Uee_x[Nei][Nej] += (Ua_xe - Ub_xe)*(-e/Q)
                        Uee_y[Nei][Nej] += (Ua_ye - Ub_ye)*(-e/Q)
                        Uee_z[Nei][Nej] += (Ua_ze - Ub_ze)*(-e/Q)
                        U_ee[Nei][Nej] = np.sqrt(np.power(Uee_x[Nei][Nej], 2) + np.power(Uee_y[Nei][Nej], 2) + np.power(Uee_x[Nei][Nej], 2))/e

                        #resultant of velocity of electrons
                        v0_e_resultant = np.sqrt(np.power(vx_electron[Nei] + vx_electron[Nej],2) + np.power(vy_electron[Nei] + vy_electron[Nej],2) + np.power(vz_electron[Nei] + vz_electron[Nej],2))

                        #velocity of the electron in electrostatic force between electron particles
                        vx_electron[Nei] += (0.5*(xi_acceleration + xj_acceleration)*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
                        vy_electron[Nei] += (0.5*(yi_acceleration + yj_acceleration)*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
                        vz_electron[Nei] += (0.5*(zi_acceleration + zj_acceleration)*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))

                        #final position of the electrons
                        x_electron[Nei] += (0.5*(vx_electron[Nei] + vx_electron[Nej])*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
                        y_electron[Nei] += (0.5*(vx_electron[Nei] + vx_electron[Nej])*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))
                        z_electron[Nei] += (0.5*(vx_electron[Nei] + vx_electron[Nej])*dt)/np.sqrt(1 - np.power(v0_e_resultant/c0,2))

                    elif Nei == Nej:
                        break

    #for the proton - proton interaction
    for Npi in range(num_particles[1]):

        ax.cla()
        
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

                    Ua_xp = U1px1 - U1px2
                    Ua_yp = U1py1 - U1py2
                    Ua_zp = U1pz1 - U1pz2

                    Ub_xp = U2px1 - U2px2
                    Ub_yp = U2py1 - U2py2
                    Ub_zp = U2pz1 - U2pz2

                    #acceleration of the proton in electrostatic force between proton particles
                    xi_acceleration = (e/me)*(Ex_field[1][grid_xip] + (0.5*((vy_proton[Npi]*Bz_field[1][grid_zip]) - (vz_proton[Npi]*By_field[1][grid_yip]))))
                    yi_acceleration = (e/me)*(Ey_field[1][grid_yip] + (0.5*((vz_proton[Npi]*Bx_field[1][grid_xip]) - (vx_proton[Npi]*Bz_field[1][grid_zip]))))
                    zi_acceleration = (e/me)*(Ez_field[1][grid_zip] + (0.5*((vx_proton[Npi]*By_field[1][grid_yip]) - (vy_proton[Npi]*Bx_field[1][grid_xip]))))

                    xj_acceleration = (e/me)*(Ex_field[1][grid_xjp] + (0.5*((vy_proton[Npj]*Bz_field[1][grid_zjp]) - (vz_proton[Npj]*By_field[1][grid_yjp]))))
                    yj_acceleration = (e/me)*(Ey_field[1][grid_yjp] + (0.5*((vz_proton[Npj]*Bx_field[1][grid_xjp]) - (vx_proton[Npj]*Bz_field[1][grid_zjp]))))
                    zj_acceleration = (e/me)*(Ez_field[1][grid_zjp] + (0.5*((vx_proton[Npj]*By_field[1][grid_yjp]) - (vy_proton[Npj]*Bx_field[1][grid_xjp]))))

                    if not(Npi == Npj):

                        #Electric Potential Energy between proton particles
                        Upp_x[Npi][Npj] += (Ua_xp - Ub_xp)*(e/Q)
                        Upp_y[Npi][Npj] += (Ua_yp - Ub_yp)*(e/Q)
                        Upp_z[Npi][Npj] += (Ua_zp - Ub_zp)*(e/Q)
                        U_pp[Npi][Npj] = np.sqrt(np.power(Upp_x[Npi][Npj], 2) + np.power(Upp_y[Npi][Npj], 2) + np.power(Upp_x[Npi][Npj], 2))/e
                        print(f"The Electric Potential Between protons: {U_pp[Npi][Npj]}")

                        #resultant of velocity of proton
                        v0_p_resultant = np.sqrt(np.power(vx_proton[Npi] + vx_proton[Npj],2) + np.power(vy_proton[Npi] + vy_proton[Npj],2) + np.power(vz_proton[Npi] + vz_proton[Npj],2))
                        
                        #velocity of the proton in electrostatic force between proton particles
                        vx_proton[Npi] += (0.5*(xi_acceleration + xj_acceleration)*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
                        vy_proton[Npi] += (0.5*(yi_acceleration + yj_acceleration)*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
                        vz_proton[Npi] += (0.5*(zi_acceleration + zj_acceleration)*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))

                        #final position of the protons
                        x_proton[Npi] += (0.5*(vx_proton[Npi] + vx_proton[Npj])*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
                        y_proton[Npi] += (0.5*(vy_proton[Npi] + vy_proton[Npj])*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))
                        z_proton[Npi] += (0.5*(vz_proton[Npi] + vz_proton[Npj])*dt)/np.sqrt(1 - np.power(v0_p_resultant/c0,2))

                    elif Npi == Npj:
                        break

    #for the proton - electron interaction
    for Ne in range(num_particles[0]):

        ax.cla()

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

                    U_xp = Upx1 - Upx2
                    U_yp = Upy1 - Upy2
                    U_zp = Upz1 - Upz2

                    U_xe = Uex1 - Uex2
                    U_ye = Uey1 - Uey2
                    U_ze = Uez1 - Uez2

                    #Electric Potential Energy between proton and electron
                    Upe_x[Ne][Np] += (U_xp - U_xe)*(e/Q)
                    Upe_y[Ne][Np] += (U_yp - U_ye)*(e/Q)
                    Upe_z[Ne][Np] += (U_zp - U_ze)*(e/Q)
                    U_pe[Ne][Np] = np.sqrt(np.power(Upe_x[Ne][Np], 2) + np.power(Upe_y[Ne][Np], 2) + np.power(Upe_z[Ne][Np], 2))/e
                    print(f"The Electric Potential Between proton and electron: {U_pe[Ne][Np]}")
    

                    #acceleration of the particle in electrostatic force between electrode surface area and charged particle
                    xe_acceleration = (-e/me)*(Ex_field[0][grid_xe] + (0.5*((vy_proton[Ne]*Bz_field[0][grid_ze]) - (vz_proton[Ne]*By_field[0][grid_ye]))))
                    ye_acceleration = (-e/me)*(Ey_field[0][grid_ye] + (0.5*((vz_proton[Ne]*Bx_field[0][grid_xe]) - (vx_proton[Ne]*Bz_field[0][grid_ze]))))
                    ze_acceleration = (-e/me)*(Ez_field[0][grid_ze] + (0.5*((vx_proton[Ne]*By_field[0][grid_ye]) - (vy_proton[Ne]*Bx_field[0][grid_xe]))))

                    xp_acceleration = (e/me)*(Ex_field[1][grid_xp] + (0.5*((vy_proton[Np]*Bz_field[1][grid_zp]) - (vz_proton[Np]*By_field[1][grid_yp]))))
                    yp_acceleration = (e/me)*(Ey_field[1][grid_yp] + (0.5*((vz_proton[Np]*Bx_field[1][grid_xp]) - (vx_proton[Np]*Bz_field[1][grid_zp]))))
                    zp_acceleration = (e/me)*(Ez_field[1][grid_zp] + (0.5*((vx_proton[Np]*By_field[1][grid_yp]) - (vy_proton[Np]*Bx_field[1][grid_xp]))))

                    #resultant of velocity of proton
                    v0_pe_resultant = np.sqrt(np.power(vx_electron[Ne] + vx_proton[Np],2) + np.power(vy_electron[Ne] + vy_proton[Np],2) + np.power(vz_electron[Ne] + vz_proton[Np],2))
                        
                    
                    #velocity of the elctron in electrostatic force between electron and proton
                    vx_electron[Ne] += (0.5*(xp_acceleration + xe_acceleration)*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))
                    vy_electron[Ne] += (0.5*(yp_acceleration + ye_acceleration)*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))
                    vz_electron[Ne] += (0.5*(zp_acceleration + ze_acceleration)*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))
                    
                    #velocity of the proton in electrostatic force between electron and proton
                    vx_proton[Np] += (0.5*(xp_acceleration + xe_acceleration)*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))
                    vy_proton[Np] += (0.5*(yp_acceleration + ye_acceleration)*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))
                    vz_proton[Np] += (0.5*(zp_acceleration + ze_acceleration)*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))

                    #final position of the electrons
                    x_electron[Ne] += (0.5*(vx_proton[Np] + vx_electron[Ne])*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))
                    y_electron[Ne] += (0.5*(vy_proton[Np] + vy_electron[Ne])*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))
                    z_electron[Ne] += (0.5*(vz_proton[Np] + vz_electron[Ne])*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))
                    
                    #final position of the proton
                    x_proton[Np] += (0.5*(vx_proton[Np] + vx_electron[Ne])*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))
                    y_proton[Np] += (0.5*(vy_proton[Np] + vy_electron[Ne])*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))
                    z_proton[Np] += (0.5*(vz_proton[Np] + vz_electron[Ne])*dt)/np.sqrt(1 - np.power(v0_pe_resultant/c0,2))

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
    
    #X.append(x_particles)
    #Y.append(y_particles)
    #Z.append(z_particles) np.linspace(0, 1*np.power(10, 12))
    
    # Plot particle positions
    electron_position = ax.scatter(z_electron, y_electron, x_electron, s = 10, marker = 'o', color = "blue")
    proton_position = ax.scatter(z_proton, y_proton, x_proton, s = 10, marker = 'o', color = "red")
    
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
    
    plt.pause(dt)

plt.show()
