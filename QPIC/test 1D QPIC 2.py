import numpy as np
import matplotlib.pyplot as plt

# Constants
h_bar = 1.0  # Reduced Planck constant
m = 1.0      # Mass of the particle
L = 10.0     # Length of the simulation domain
N = 100      # Number of grid points
dx = L / N   # Grid spacing

# Time parameters
dt = 0.01    # Time step
t_max = 10   # Maximum time

# Initialize wavefunction
x = np.linspace(0, L, N)
psi = np.exp(-(x - L/2)**2) * np.exp(1j * 0.1 * x)  # Initial wavefunction

# Plot initial wavefunction
plt.figure(figsize=(8, 6))
plt.plot(x, np.real(psi), label='Real part')
plt.plot(x, np.imag(psi), label='Imaginary part')
plt.title('Initial Wavefunction')
plt.xlabel('Position')
plt.ylabel('Wavefunction')
plt.legend()
plt.grid()
plt.show()

# Time evolution using PIC method
t = 0
while t < t_max:
    # Time evolution using Schrödinger equation (free particle)
    psi -= 1j * h_bar * dt / (2 * m * dx**2) * (np.roll(psi, 1) - 2 * psi + np.roll(psi, -1))

    # Boundary conditions (periodic)
    psi[0] = psi[N - 1]
    psi[N - 1] = psi[0]

    # Plot wavefunction at certain time intervals
    if t % 1 == 0:
        plt.figure(figsize=(8, 6))
        plt.plot(x, np.real(psi), label='Real part')
        plt.plot(x, np.imag(psi), label='Imaginary part')
        plt.title(f'Wavefunction at time t = {t}')
        plt.xlabel('Position')
        plt.ylabel('Wavefunction')
        plt.legend()
        plt.grid()
        plt.show()

    t += dt
