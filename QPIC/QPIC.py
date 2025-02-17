import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0     # Particle mass
dt = 0.01   # Time step

# Potential function (modify as needed)
def potential(x):
    return 0.5 * m * omega**2 * x**2

# Initial wave function (modify as needed)
def initial_wave_function(x):
    return np.exp(-0.5 * (x - x0)**2 / sigma**2) * np.exp(1j * k0 * x)

# Time-stepping function (modify as needed)
def time_step(psi, potential):
    psi = psi * np.exp(-1j * potential * dt / hbar)
    psi_fft = np.fft.fft(psi)
    psi_fft *= np.exp(-1j * hbar * k**2 * dt / (2 * m))
    psi = np.fft.ifft(psi_fft)
    return psi

# Simulation parameters
omega = 1.0     # Oscillator frequency
x0 = -2.0       # Initial position
sigma = 0.5     # Initial width
k0 = 5.0        # Initial wavenumber

# Grid parameters
x_min, x_max = -5.0, 5.0
N = 512         # Number of grid points
dx = (x_max - x_min) / (N - 1)
x = np.linspace(x_min, x_max, N)
k = 2 * np.pi * np.fft.fftfreq(N, dx)

# Initialize wave function
psi = initial_wave_function(x)

# Main loop
num_steps = 100
for step in range(num_steps):
    # Calculate potential
    potential_energy = potential(x)
    
    # Update wave function
    psi = time_step(psi, potential_energy)

    # Plot results (modify as needed)
    if step % 10 == 0:
        plt.plot(x, np.abs(psi)**2, label=f'Step {step}')

plt.title('Quantum Particle-in-Cell Method')
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
