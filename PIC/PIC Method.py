import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure and axis for animation
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Initialize the plot for the moving point
point, = ax.plot([], [], 'bo', markersize=10)

# Initial position of the point
initial_position = [0, 0]

# Update function for animation
def update(frame):
    # Calculate new position based on time/frame
    new_x = initial_position[0] + frame * 0.1
    new_y = initial_position[1] + np.sin(frame * 0.5)
    
    # Update point position in the plot
    point.set_data(new_x, new_y)
    
    return point,

# Create the animation
ani = FuncAnimation(fig, update, frames=100, blit=True)

# Show the animation
plt.show()
