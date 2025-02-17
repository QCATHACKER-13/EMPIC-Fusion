import numpy as np
import matplotlib.pyplot as plt

# constant
epsilon_0 = 8.854187817e-12 #permittivity of free space
k0 = 1/(4*np.pi*epsilon_0)
e = 1.60217663e-19 # elementary charge
h_bar = 1.054571628e-34 #reduced plank's constant (J*s)
h = 6.62606896e-34 # plank's constant (J*s)
mu_0 = 1.256637061e-6 # permeability of free space
mp = 1.672621637e-27 # kg, mass of a proton
mn = 1.674927211e-27 # kg, mass of a neutron
me = 9.10938215e-31 # kg, mass of an electron
c0 = 299792458 # m/s, speed of light
alpha = k0*(np.power(e,2)/(h_bar*c0)) #fine tune constant
r0_mim = 1.2e-15*np.cbrt(2) # minimum nuclear radius parameter
r0_max = np.power(2*np.power(e, 2)*np.sqrt(k0)*(np.sqrt(mp)/h_bar), 2)
mH = 1.6735575e-27 # kg, mass of hydrogen atom
density_H = 0.0899 # kg/m^3, density of hydrogen
Up = 2*((me*mp)/(me + mp))*np.power(c0,2) #potential of proton

# Potential energy function (e.g., harmonic oscillator)
def potential_energy(x):
    omega = 3.31*np.power(10, 16)  # Angular frequency of the oscillator
    return 0.5 * me * omega**2 * x**2

# Quantum Particle class
class QuantumParticle:
    def __init__(self, x, p):
        self.x = x  # Initial position
        self.p = p  # Initial momentum

# Quantum Particle-in-Cell simulation
def quantum_pic(num_particles, num_steps, dt):
    particles = [QuantumParticle(np.random.rand(), np.random.rand()) for _ in range(num_particles)]

    for step in range(num_steps):
        for particle in particles:
            # Update momentum based on potential energy gradient
            particle.p -= potential_energy(particle.x) * dt / h_bar

            # Update position based on momentum
            particle.x += (particle.p / me) * dt

            # Apply periodic boundary conditions
            particle.x %= 1.0

    return [particle.x for particle in particles]

# Parameters
num_particles = 100
num_steps = 1000
dt = 0.01

# Run the simulation
final_positions = quantum_pic(num_particles, num_steps, dt)

# Plot the final positions
plt.hist(final_positions, bins=30, density=True, alpha=0.7, color='b', edgecolor='black')
plt.title('Quantum Particle-in-Cell Simulation (1D)')
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.show()

 