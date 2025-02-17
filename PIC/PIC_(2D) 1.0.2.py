import numpy as np
import matplotlib.pyplot as plt

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
I = np.pi*Q*np.power(0.5e-2,2)
dt = 1e-3 # time step

# Number of grid points
Nx = 100
Ny = 100

# parameters for the particles
num_particles = 2

# Electric field
Ex_field = np.zeros(Nx)
Ey_field = np.zeros(Ny)

# x,y particle initial position
x_particles = np.random.uniform(0, l, num_particles)
y_particles = np.random.uniform(-b, b, num_particles)

# Initial x,y particle velocity
x_velocity = np.random.uniform(0, x_particles/dt, num_particles)
y_velocity = np.random.uniform(0, y_particles/dt, num_particles)


#Electric Field
def E(q, a1, X, Y, Z):
    r = np.sqrt(np.power(X-a1[0],2) + np.power(Y - a1[1],2) + np.power(Z - a1[2],2))
    E_x = 2*k0*q*(X-a1[0])/np.power(r,3)
    E_y = k0*q*(Y-a1[1])/np.power(r,3)
    E_z = k0*q*(Z-a1[2])/np.power(r,3)
    return E_x,E_y,E_z

# Main simulation loop
for step in range(1000):
    
    # Clear electric field
    Ex_field[:] = 0.0
    Ey_field[:] = 0.0
    
    # Calculate electric field due to particles' charges and positions
    for particle in range(num_particles):
        
        x = x_particles[particle]/(l * Nx)
        y = y_particles[particle]/(b * Ny)
        
        grid_x = int(x)
        grid_y = int(np.abs(y))
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny):

            # vector function of an electric field
            Ex1, Ey1, Ez1 = E(Q, [0, 0, 0], x, 0, 0)
            Ex2, Ey2, Ez2 = E(-Q, [l, 0, 0], x, 0, 0)

            # x,y of an electric field
            Ex_field[grid_x] += Ex1 + Ex2
            Ey_field[grid_y] += Ey1 + Ey2
    
    # Update particle positions and velocities
    for particle in range(num_particles):

        x = x_particles[particle]/(l * Nx)
        y = y_particles[particle]/(b* Ny)
        
        grid_x = int(x)
        grid_y = int(np.abs(y))
        
        if (0 <= grid_x < Nx) and (0 <= grid_y < Ny):

            #acceleration of the particle in electrostatic force between electrode surface area and charged particle
            x_acceleration = (e/me)*Ex_field[grid_x]
            y_acceleration = (e/me)*Ey_field[grid_y]

            #velocity of the particle in electrostatic force between electrode surface area and charged particle
            x_velocity[particle] += x_acceleration*dt
            y_velocity[particle] += y_acceleration*dt

            #final position of the particle
            x_particles[particle] += x_velocity[particle]*dt
            y_particles[particle] += y_velocity[particle]*dt
    
    # Periodic boundary conditions
    x_particles = np.mod(x_particles - a, l - 2*a) + a
    y_particles = np.mod(y_particles, y_particles + 2*b) - b
    
    # Plot particle positions
    plt.clf()
    plt.plot(x_particles, y_particles, 'ro')
    plt.xlim(0, l)
    plt.ylim(-b, b)
    plt.grid()
    plt.xlabel("Position")
    plt.title("Particle Motion with Interaction")
    plt.pause(dt)

# Visualization or data analysis can be added here
plt.show()
