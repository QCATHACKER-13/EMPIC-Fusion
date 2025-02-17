import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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
a = 0.5e-2 # center gap distance between two elctrodes
b = 0.15e-2 # radii of the electrode
d = 1.0e-2 # diameter of the cylinder glass
l = 2.50e-2 # lenght of the cylinder glass
h = l/2 # half of the lenght of cylindrical glass
Q = 2.25e-6 # Coulumb
N = 47875.72074 #total of turns = no. of turns in lenght * no. of turn
I = np.pi*Q*np.power(0.5e-2,2)
dt = 1e-3 # time step

#Identical Path of the particle
X = []
Y = []
Z = []

# Number of grid points
Nx = 100
Ny = 100
Nz = 100

# parameters for the particles
num_particles = 1

# Electric field
Ex_field = np.zeros(Nx)
Ey_field = np.zeros(Ny)
Ez_field = np.zeros(Nz)

# Magnetic field
B1x_field = np.zeros(Nx)
B1y_field = np.zeros(Ny)
B1z_field = np.zeros(Nz)

B2x_field = np.zeros(Nx)
B2y_field = np.zeros(Ny)
B2z_field = np.zeros(Nz)

# x,y particle initial position
x_particles = np.random.uniform(-b, b, num_particles)
y_particles = np.random.uniform(-b, b, num_particles)
z_particles = np.random.uniform(a, l - a, num_particles)

# Initial x,y particle velocity
x_velocity = np.random.uniform(0, x_particles/dt, num_particles)
y_velocity = np.random.uniform(0, y_particles/dt, num_particles)
z_velocity = np.random.uniform(0, z_particles/dt, num_particles)

# Create the figure and axes
fig = plt.figure(dpi=100)
ax = fig.add_subplot(projection='3d')

#Electric Field
def E(q, a1, X, Y, Z):
    r = np.sqrt(np.power(X-a1[0],2) + np.power(Y - a1[1],2) + np.power(Z - a1[2],2))
    E_x = 2*k0*q*(X-a1[0])/np.power(r,3)
    E_y = k0*q*(Y-a1[1])/np.power(r,3)
    E_z = k0*q*(Z-a1[2])/np.power(r,3)
    return E_x,E_y,E_z

def B1(I0, a3, X, Y, Z):
    r = np.sqrt(np.power(X-a3[0],2) + np.power(Y - a3[1],2) + np.power(Z - a3[2],2))
    B_x1 = (mu_0*I0*(Y - a3[1]))/(2*np.pi*np.power(r,3))
    B_y1 = -(mu_0*I0*(X - a3[0]))/(2*np.pi*np.power(r,3))
    B_z1 = np.zeros_like(r)
    return B_x1,B_y1,B_z1

def B2(I0, a4, X, Y, Z):
    r = np.sqrt(4*np.power(X-a4[0],2) + 4*np.power(Y - a4[1],2) + np.power(Z - a4[2],2))
    B_x2 = (mu_0*I0*N*(X - a4[0]))/(2*np.pi*np.power(r,3))
    B_y2 = (mu_0*I0*N*(Y - a4[1]))/(2*np.pi*np.power(r,3))
    B_z2 = (mu_0*I0*N*(Z - a4[2]))/(2*np.pi*np.power(r,3))
    return B_x2,B_y2,B_z2

# Main simulation loop
for step in range(1000):
    
    # Clear electric field
    Ex_field[:] = 0.0
    Ey_field[:] = 0.0
    Ez_field[:] = 0.0

    # Clear magnetic field
    B1x_field[:] = 0.0
    B1y_field[:] = 0.0
    B2x_field[:] = 0.0
    B2y_field[:] = 0.0
    
    # Calculate electric field due to particles' charges and positions
    for particle in range(num_particles):
        
        x = x_particles[particle]/(b * Nx)
        y = y_particles[particle]/(b * Ny)
        z = z_particles[particle]/(l * Nz)
        
        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 < grid_z <= Nz):

            # vector function of an electric field
            Ex1, Ey1, Ez1 = E(Q, [0, 0, a], x, y, z)
            Ex2, Ey2, Ez2 = E(-Q, [0, 0, l -a], x, y, z)

            # x,y of an electric field
            Ex_field[grid_x] += Ex1 + Ex2
            Ey_field[grid_y] += Ey1 + Ey2
            Ez_field[grid_z] += Ez1 + Ez2

            #calculate vector function of magnetic field perpendicular to the electric field
            Bx1_1, By1_1, Bz1_1 = B1(I, [0, 0, a], x, y, z)
            Bx2_1, By2_1, Bz2_1 = B1(I, [0, 0, l -a], x, y, z)
            Bx1 = Bx1_1 + Bx2_1
            By1 = By1_1 + By2_1
            Bz1 = Bz1_1 + Bz2_1

            #calculate vector function of magnetic field of the electromagnet
            Bx1_2, By1_2, Bz1_2 = B2(I, [0, 0, a], x, y, z)
            Bx2_2, By2_2, Bz2_2 = B2(I, [0, 0, l -a], x, y, z)
            Bx2 = Bx1_2 - Bx2_2
            By2 = By1_2 - By2_2
            Bz2 = Bz1_2 - Bz2_2

            #calculate the B1 addition of B2
            B1x_field[grid_x] += Bx1
            B2x_field[grid_x] += Bx2
            B1y_field[grid_y] += By1
            B2y_field[grid_y] += By2

            print(f"({step}) steps: {Ex_field[grid_x],Ey_field[grid_y]}:")
    
    # Update particle positions and velocities
    for particle in range(num_particles):
        ax.cla()

        x = x_particles[particle]/(b * Nx)
        y = y_particles[particle]/(b * Ny)
        z = z_particles[particle]/(l * Nz)

        print(f"({step}) steps: {x,y}:")
        grid_x = int(np.abs(x))
        grid_y = int(np.abs(y))
        grid_z = int(z)
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny) and (0 < grid_z <= Nz):

            #acceleration of the particle in electrostatic force between electrode surface area and charged particle
            x_acceleration = (e/mp)*(Ex_field[grid_x] + (0.5*x_velocity[particle]*(B1x_field[grid_x] + B2x_field[grid_x])))
            y_acceleration = (e/mp)*(Ey_field[grid_y] + (0.5*y_velocity[particle]*(B1y_field[grid_y] + B2y_field[grid_y])))
            z_acceleration = (e/mp)*(Ez_field[grid_z] + (0.5*z_velocity[particle]*(B1z_field[grid_z] + B2z_field[grid_z])))

            #velocity of the particle in electrostatic force between electrode surface area and charged particle
            x_velocity[particle] -= (0.5*x_acceleration*dt)/np.sqrt(1 - np.power((x_acceleration*dt)/c0,2))
            y_velocity[particle] -= (0.5*y_acceleration*dt)/np.sqrt(1 - np.power((y_acceleration*dt)/c0,2))
            z_velocity[particle] -= (0.5*z_acceleration*dt)/np.sqrt(1 - np.power((z_acceleration*dt)/c0,2))

            #final position of the particle
            x_particles[particle] -= (x_velocity[particle]*dt)/np.sqrt(1 - np.power((x_acceleration*dt)/c0,2))
            y_particles[particle] -= (y_velocity[particle]*dt)/np.sqrt(1 - np.power((y_acceleration*dt)/c0,2))
            z_particles[particle] -= (z_velocity[particle]*dt)/np.sqrt(1 - np.power((z_acceleration*dt)/c0,2))
    
    # Periodic boundary conditions
    x_particles = np.mod(x_particles, x_particles + 2*b) - b
    y_particles = np.mod(y_particles, y_particles + 2*b) - b
    z_particles = np.mod(z_particles - a, l - 2*a) + a

    X.append(x_particles)
    Y.append(y_particles)
    Z.append(z_particles)
    
    # Plot particle positions
    ax.scatter(z_particles, y_particles, x_particles, s = 100, marker = 'o')
    #ax.plot(Z, Y, X)
    ax.set_xlim(0, l)
    ax.set_ylim(-b, b)
    ax.set_zlim(-b, b)
    ax.set_xlabel("z - Position")
    ax.set_ylabel("y - Position")
    ax.set_zlabel("x - Position")
    ax.set_title("Particle Motion with Interaction")
    plt.pause(dt)

ani = FuncAnimation(fig = fig, func = update, frames = 100, interval = 100, repeat = False)

# Visualization or data analysis can be added here
plt.show()
