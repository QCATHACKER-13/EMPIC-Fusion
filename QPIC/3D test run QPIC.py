import numpy as np
import matplotlib.pyplot as plt

# Constants
dt = 0.01  # Time step
dx = 0.1   # Spatial step

# Particle properties
num_particles = 100
charge = -1.0
mass = 1.0

# Grid properties
num_grid_points = 200
grid = np.linspace(0, num_grid_points * dx, num_grid_points)
fields = np.zeros_like(grid)

# Initialize particle positions and wavefunctions
positions = np.random.rand(num_particles) * num_grid_points * dx
wavefunctions = np.exp(-(grid - positions[:, np.newaxis])**2 / (2 * dx**2))

# Main simulation loop
num_steps = 100

for step in range(num_steps):
    # Update wavefunctions based on the Schrödinger equation
    wavefunctions *= np.exp(-1j * charge * fields * dt / (2 * mass))

    # Calculate charge density from the wavefunctions
    charge_density = np.abs(wavefunctions)**2

    # Solve for the electric field based on the charge density
    fields = -np.gradient(charge_density, dx)

    # Update particle positions using the fields
    positions += dt * np.gradient(np.angle(wavefunctions), dx, axis=1)

    # Update wavefunctions based on the new positions
    wavefunctions = np.exp(-(grid - positions[:, np.newaxis])**2 / (2 * dx**2))

    # Perform quantum measurements or additional diagnostics if needed

# Plot the final results
plt.plot(grid, charge_density.sum(axis=0))
plt.title('Charge Density Evolution')
plt.xlabel('Position')
plt.ylabel('Charge Density')
plt.show()
