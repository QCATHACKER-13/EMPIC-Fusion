import numpy as np
import matplotlib.pyplot as plt

# Constants
h_bar = 1.0  # Reduced Planck constant
mass = 1.0  # Particle mass
L = 10.0  # Length of the simulation box
num_points = 1000  # Number of spatial points
dt = 0.01  # Time step
num_steps = 100  # Number of time steps

# Spatial grid
x = np.linspace(0, L, num_points, endpoint=False)

# Initial wavefunction parameters
k0 = 2 * np.pi / L  # Initial wavenumber
x0 = L / 4  # Initial position
sigma = 0.2 * L  # Width of the wavepacket

# Initial wavefunction
psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.trapz(np.abs(psi)**2, x))  # Normalize the wavefunction

# Time evolution using the Schrödinger equation
for t in range(num_steps):
    # Kinetic energy term
    psi_kinetic = np.fft.fft(psi)
    psi_kinetic *= np.exp(-1j * h_bar * k0**2 * dt / (2 * mass) * (2 * np.pi / L * np.arange(num_points))**2)
    psi = np.fft.ifft(psi_kinetic)

    # Potential energy term (free particle has no potential energy)

    # Normalize the wavefunction
    psi /= np.sqrt(np.trapz(np.abs(psi)**2, x))

    # Plot the wavefunction at certain time steps
    if t % 10 == 0:
        plt.plot(x, np.abs(psi)**2, label=f'Time Step {t}')

plt.title('Time Evolution of the Wavefunction for a Free Particle')
plt.xlabel('Position (x)')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
