import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0     # Particle mass
dt = 0.01   # Time step

# Potential function (modify as needed)
def potential(x, y, z):
    return 0.5 * m * (omega_x**2 * x**2 + omega_y**2 * y**2 + omega_z**2 * z**2)

# Initial wave function (modify as needed)
def initial_wave_function(x, y, z):
    return np.exp(-0.5 * ((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / sigma**2) * np.exp(1j * (k0x * x + k0y * y + k0z * z))

# Time-stepping function (modify as needed)
def time_step(psi, potential):
    psi = psi * np.exp(-1j * potential * dt / hbar)
    psi_fft = np.fft.fftn(psi)
    psi_fft *= np.exp(-1j * hbar * (kx**2 + ky**2 + kz**2) * dt / (2 * m))
    psi = np.fft.ifftn(psi_fft)
    return psi

# Simulation parameters
omega_x, omega_y, omega_z = 1.0, 1.0, 1.0  # Oscillator frequencies
x0, y0, z0 = -2.0, -2.0, -2.0                # Initial positions
sigma = 0.5                                 # Initial width
k0x, k0y, k0z = 5.0, 5.0, 5.0                # Initial wavenumbers

# Grid parameters
x_min, x_max = -5.0, 5.0
y_min, y_max = -5.0, 5.0
z_min, z_max = -5.0, 5.0
N = 64  # Number of grid points in each dimension
dx = (x_max - x_min) / (N - 1)
dy = (y_max - y_min) / (N - 1)
dz = (z_max - z_min) / (N - 1)
x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, N)
z = np.linspace(z_min, z_max, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
kx = 2 * np.pi * np.fft.fftfreq(N, dx)
ky = 2 * np.pi * np.fft.fftfreq(N, dy)
kz = 2 * np.pi * np.fft.fftfreq(N, dz)

# Initialize wave function
psi = initial_wave_function(X, Y, Z)

# Main loop
num_steps = 50
for step in range(num_steps):
    # Calculate potential
    potential_energy = potential(X, Y, Z)
    
    # Update wave function
    psi = time_step(psi, potential_energy)

    # Plot results (modify as needed)
    if step % 5 == 0:
        plt.figure()
        plt.imshow(np.abs(psi[:, :, N // 2])**2, extent=(x_min, x_max, y_min, y_max), origin='lower')
        plt.title(f'Quantum Particle-in-Cell Method - Step {step}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.show()
