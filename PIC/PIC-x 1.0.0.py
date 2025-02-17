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
Nx = 100       # Number of grid points
dt = 1e-3 #time step

#parameters for the particles
num_particles = 1
E_field = np.zeros(Nx)  # Electric field
particles = np.random.uniform(0, l, num_particles)  # Initial particle positions
velocities = np.random.normal(0, 1, num_particles)  # Initial particle velocities

# Create the figure and axes
fig = plt.figure(figsize=(5,4), dpi=80)

# Function to calculate electric field
def calculate_electric_field():
    E_field[:] = 0.0
    for particle in range(num_particles):
        grid_x = int(particles[particle] / l * Nx)
        if 0 <= grid_x < Nx:
            E_field[grid_x] += (2*k0*Q/np.power(grid_x,2))

# Recursive function to update particle positions and velocities
def update_particles(particle_index):
    if particle_index < num_particles:
        calculate_electric_field()
        grid_x = int(particles[particle_index] / l * Nx)
        if 0 <= grid_x < Nx:
            acceleration = (-e/me) * E_field[grid_x]
            velocities[particle_index] += acceleration * dt
            particles[particle_index] += velocities[particle_index] * dt
        particles[particle_index] = np.mod(particles[particle_index], l)
        update_particles(particle_index + 1)

# Main simulation loop
for step in range(1000):
    update_particles(0)
    
    # Plot particle positions
    plt.clf()
    plt.plot(particles, np.zeros(num_particles), 'ro')
    plt.xlim(0, l)
    plt.ylim(-0.1, 0.1)
    plt.xlabel('Position')
    plt.title('Particle Motion with Interaction (Recursion)')
    plt.pause(dt)

# Visualization or data analysis can be added here
plt.show()
