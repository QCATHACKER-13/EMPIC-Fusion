import numpy as np
import matplotlib.pyplot as plt

# Step 1: Initialize Parameters
dt = 0.01
N_particles = 100
T = 1.0
L = 10.0

# Electric potential function (modify as needed)
def electric_potential(x):
    return -np.sin(2 * np.pi * x / L)

# Step 2: Initialize Particle Properties
x_particles = np.random.uniform(0, L, N_particles)
v_particles = np.zeros(N_particles)

# Main Loop
for t in np.arange(0, T, dt):
    # Step 3: Compute Charge Density
    charge_density = np.histogram(x_particles, bins=np.linspace(0, L, 100))[0]

    # Step 4: Solve Poisson's Equation
    electric_potential_values = -np.cumsum(charge_density) * (L / len(charge_density))

    # Step 5: Compute Electric Field
    electric_field = -np.gradient(electric_potential_values, dx = L/len(charge_density))

    # Step 6: Update Particle Velocities (Leapfrog Method)
    v_particles += (electric_field[np.searchsorted(np.linspace(0, L, len(electric_field)), x_particles)] / m) * dt / 2

    # Step 7: Update Particle Positions
    x_particles += v_particles * dt

# Plot the final particle positions
plt.scatter(x_particles, np.zeros(N_particles))
plt.title('Particle Positions')
plt.xlabel('Position')
plt.ylabel('Particle')
plt.show()
