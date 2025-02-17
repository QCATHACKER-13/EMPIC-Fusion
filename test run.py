import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

fig = plt.figure()
ax = fig.add_subplot(projection = "3d")

# Create a scatter plot
ax.scatter(x, y, z)
ax.set_title('Scatter Plot')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()
