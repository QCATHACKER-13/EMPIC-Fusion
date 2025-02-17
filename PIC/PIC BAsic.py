import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 10.0       # Simulation domain size
Nx = 100       # Number of grid points
dt = 0.01      # Time step
num_particles = 1
charge = -1.0 # Electron charge
mass = 1.0    # Electron mass
E_field = np.zeros(Nx)  # Electric field
B_field = 1.0           # Magnetic field strength
particles = np.random.uniform(0, L, num_particles)  # Initial particle positions
velocities = np.random.normal(0, 1, num_particles)  # Initial particle velocities

# Main simulation loop
for step in range(1000):
    # Clear electric field
    E_field[:] = 0.0
    
    # Calculate electric field (in this example, it's a constant field)
    E_field[:] = 1.0
    
    # Update particle positions and velocities
    for particle in range(num_particles):
        grid_x = int(particles[particle] / L * Nx)
        if 0 <= grid_x < Nx:
            acceleration = (charge / mass) * (E_field[grid_x] + velocities[particle] * B_field)
            velocities[particle] += acceleration * dt
            particles[particle] += velocities[particle] * dt
    
    # Periodic boundary conditions
    particles = np.mod(particles, L)
    
    # Plot particle positions
    plt.clf()
    plt.plot(particles, np.zeros(num_particles), 'ro')
    plt.xlim(0, L)
    plt.ylim(-0.1, 0.1)
    plt.xlabel('Position')
    plt.title('Particle Motion in Electric and Magnetic Fields')
    plt.pause(0.01)

# Visualization or data analysis can be added here
plt.show()
