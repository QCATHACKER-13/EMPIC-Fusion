import matplotlib.pyplot as plt
import numpy as np
import random

# Simulation parameters
num_particles = 100
num_steps = 100
dt = 0.01

# Particle properties
charge = -1.0
mass = 1.0

# Grid properties
grid_size = 100
dx = 1.0
dy = 1.0

x = []
y = []
velocities_x = np.zeros(num_particles)
velocities_y = np.zeros(num_particles)
electric_fieldx = np.zeros(grid_size)
electric_fieldy = np.zeros(grid_size)
magnetic_fieldx = np.zeros(grid_size)
magnetic_fieldy = np.zeros(grid_size)


# Main simulation loop
for step in range(num_steps):
    # Calculate electric field based on particle positions
    for i in range(grid_size):
        for p in range(num_particles):
            x.append(np.random.rand(num_particles))
            y.append(np.random.rand(num_particles))
            distance_x = x[p] - (i * dx)
            distance_y = y[p] - (i * dy)
            electric_fieldx += (charge / distance_x ** 2)
            electric_fieldy += (charge / distance_y ** 2)
            
            # Push particle velocities
            acceleration_x = (charge * electric_fieldx) / mass
            acceleration_y = (charge * electric_fieldy) / mass
            velocities_x += acceleration_x * dt
            velocities_y += acceleration_y * dt
            
            # Move particles
            x += velocities_x * dt
            y += velocities_y * dt
            
            plt.xlim(0,grid_size)
            plt.ylim(0,grid_size)
            plt.scatter(x,y,color  = 'black')
            plt.pause(dt)
            
            # Reset electric field for the next step
            electric_fieldx = np.zeros(grid_size)
            electric_fieldy = np.zeros(grid_size)
        
plt.show()
