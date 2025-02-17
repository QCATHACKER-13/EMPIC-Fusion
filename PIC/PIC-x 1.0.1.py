import numpy as np
import matplotlib.pyplot as plt

#constant
epsilon_0 = 8.854187817e-12
k0 = 1/(4*np.pi*epsilon_0)
e = 1.60217663e-19
mu_0 = 1.256637061e-6
mp = 1.672621637e-27 #kg, mass of a proton
mn = 1.674927211e-27 #kg, mass of a neutron
me = 9.10938215e-31 #kg, mass of an electron
c0 = 299792458 #m/s, speed of light

#Parameters
a = 0.5e-2 #center gap distance between two elctrodes
b = 0.15e-2 #radii of the electrode
d = 1.0e-2 #diameter of the cylinder glass
l = 2.50e-2 #lenght of the cylinder glass
h = l/2 #half of the lenght of cylindrical glass
Q = 2.25e-6 #aCoulumb length=0.2, 
I = np.pi*Q*np.power(0.5e-2,2)
Nx, Ny = 100, 100       # Number of grid points
dt = 1e-3 #time step

#parameters for the particles
num_particles = 1
E_field = np.zeros(Nx)  # Electric field
particles = np.random.uniform(a, l - a, num_particles)  # Initial particle positions
velocities = np.random.normal(0, particles/dt, num_particles) # Initial particle velocities

# Main simulation loop
for step in range(1000):
    # Clear electric field
    E_field[:] = 0.0
    
    # Calculate electric field due to particles' charges and positions
    for particle in range(num_particles):
        x = particles[particle] / l * Nx
        grid_x = int(x)
        if 0 <= grid_x < Nx:
            E_field[grid_x] += (k0*Q/np.power(x, 2)) - (k0*Q/np.power(x - l, 2))
    
    # Update particle positions and velocities
    for particle in range(num_particles):
        x = particles[particle] / l * Nx
        grid_x = int(x)
        if 0 <= grid_x < Nx:
            acceleration = (-e/me) * E_field[grid_x]
            velocities[particle] += acceleration * dt
            particles[particle] += (a + velocities[particle]*dt)
    
    # Periodic boundary conditions
    if particles > a:
        particles = np.mod(particles - a, l - a) + a

        # Plot particle positions
        plt.clf()
        plt.plot(particles, np.zeros(num_particles), 'ro')

    if particle < (l - a):
        particles = np.mod(particles + a, l - a)
        
        # Plot particle positions
        plt.clf()
        plt.plot(particles, np.zeros(num_particles), 'ro')
        
    plt.xlim(0, l)
    plt.ylim(-b, b)
    plt.grid()
    plt.xlabel("Position")
    plt.title("Particle Motion with Interaction")
    plt.pause(dt)

# Visualization or data analysis can be added here
plt.show()
