import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
L = 1.0           # Length of the simulation domain
N_particles = 100  # Number of particles
dt = 0.01          # Time step
t_max = 1.0        # Maximum simulation time

# Particle properties
charge = -1.0      # Particle charge (electron)
mass = 1.0         # Particle mass (normalized to electron mass)

# Electric field
E0 = 1.0           # Electric field strength

# Initialize particle positions and velocities
x_particles = np.random.rand(N_particles) * L   # Random initial positions
v_particles = np.zeros(N_particles)              # Zero initial velocities

# Main simulation loop
t = 0.0
while t < t_max:
    # Update particle positions using velocity
    x_particles += v_particles * dt
    
    # Update particle velocities using the electric field
    v_particles += (charge / mass) * E0 * dt
    
    # Periodic boundary conditions
    x_particles = np.mod(x_particles, L)
    
    # Increment time
    t += dt

# Plot the final particle positions
plt.scatter(x_particles, np.zeros(N_particles), marker='o', label='Particles')
plt.xlabel('Position')
plt.title('Particle-in-Cell Simulation')
plt.legend()
plt.show()
