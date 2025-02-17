import numpy as np
import matplotlib.pyplot as plt

# Constants
num_particles = 100  # Number of particles
x_min = 0.5e-2       # Minimum x-coordinate
x_max = 2e-2         # Maximum x-coordinate
dt = 1e-6             # Time step
num_steps = 100      # Number of time steps

# Initialize particle positions and velocities within the specified range
x_particles = np.linspace(x_min, x_max, num_particles)
v_particles = np.zeros(num_particles)

# Arrays to store particle history for plotting
x_history = np.zeros((num_steps, num_particles))
v_history = np.zeros((num_steps, num_particles))

# Main time-stepping loop
for step in range(num_steps):

    for particle in range(num_particles):
        # Store current positions and velocities
        x_history[step, :] = x_particles
        v_history[step, :] = v_particles

        # Update particle positions using velocity
        x_particles += v_particles * dt

        # Particle boundary conditions (reflective boundaries)
        x_particles[x_particles < x_min] = 2 * x_min - x_particles[x_particles < x_min]
        x_particles[x_particles > x_max] = 2 * x_max - x_particles[x_particles > x_max]

        # Update particle velocities (assuming no electromagnetic forces)
        # In a real PIC simulation, you would compute forces and update velocities here

# Plot the particle trajectories
    
plt.plot(x_history[:, particle], v_history[:, particle])
plt.xlabel('Position (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Particle Trajectories')
plt.grid(True)
plt.show()
