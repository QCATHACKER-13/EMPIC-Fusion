import numpy as np
import matplotlib.pyplot as plt

from sgrid import Grid
from quant import Particles


domain_size = 10.0
num_cells = 100
num_particles = 1000

grid = Grid(num_cells, domain_size)
particles = Particles(num_particles, domain_size)

def deposit_charge(particles, grid):
    grid.charge_density.fill(0)
    for i in range(particles.num_particles):
        cell_index = int(particles.positions[i] // grid.cell_size)
        grid.charge_density[cell_index] += particles.charges[i] / grid.cell_size

def solve_poisson(grid):
    # Using a simple finite difference method for Poisson's equation
    rho = grid.charge_density
    phi = grid.potential
    dx = grid.cell_size

    # Simple finite difference solver (could be replaced with a more efficient solver)
    for i in range(1, grid.num_cells - 1):
        phi[i] = (phi[i-1] + phi[i+1] + dx**2 * rho[i]) / 2

    # Calculate electric field E = -dPhi/dx
    for i in range(1, grid.num_cells - 1):
        grid.electric_field[i] = -(phi[i+1] - phi[i-1]) / (2 * dx)

def interpolate_field_to_particles(particles, grid):
    particle_fields = np.zeros(particles.num_particles)
    for i in range(particles.num_particles):
        cell_index = int(particles.positions[i] // grid.cell_size)
        particle_fields[i] = grid.electric_field[cell_index]
    return particle_fields

def update_particles(particles, particle_fields, dt):
    particles.velocities += (particle_fields / particles.masses) * dt
    particles.positions += particles.velocities * dt

    # Apply periodic boundary conditions
    particles.positions %= grid.domain_size

def apply_boundary_conditions(particles, grid):
    # Periodic boundary conditions already applied in update_particles
    pass

def compute_energy(particles):
    kinetic_energy = 0.5 * np.sum(particles.masses * particles.velocities**2)
    return kinetic_energy

def log_diagnostics(timestep, particles):
    energy = compute_energy(particles)
    print(f"Timestep: {timestep}, Kinetic Energy: {energy}")

num_timesteps = 1000
dt = 0.01

for timestep in range(num_timesteps):
    deposit_charge(particles, grid)
    solve_poisson(grid)
    particle_fields = interpolate_field_to_particles(particles, grid)
    update_particles(particles, particle_fields, dt)
    apply_boundary_conditions(particles, grid)
    log_diagnostics(timestep, particles)

plt.plot(particles.positions, np.zeros_like(particles.positions), 'o')
plt.xlabel('Position')
plt.ylabel('Particles')
plt.title('Particle Positions')
plt.show()

