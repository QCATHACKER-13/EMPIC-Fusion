import numpy as np

class Particles:
    def __init__(self, num_particles, domain_size):
        self.num_particles = num_particles
        self.positions = np.random.rand(num_particles) * domain_size
        self.velocities = np.zeros(num_particles)
        self.charges = np.ones(num_particles)
        self.masses = np.ones(num_particles)
