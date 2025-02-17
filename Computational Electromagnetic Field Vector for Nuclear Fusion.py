import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Parameters
a = 0.5e-2 #center gap distance between two elctrodes
b = 0.15e-2 #radii of the electrode
d = 1.0e-2 #diameter of the cylinder glass
l = 2.50e-2 #lenght of the cylinder glass
h = l/2 #half of the lenght of cylindrical glass
Q = 2.25e-6 #aCoulumb length=0.2, 
I = np.pi*Q*np.power(0.5e-2,2)

#parameters
epsilon_0 = 8.854187817e-12
k0 = 1/(4*np.pi*epsilon_0)
e = 1.60217663e-19
mu_0 = 1.256637061e-6

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) 


# Make the grid
x, y, z = np.meshgrid(
           np.linspace(0, l, 10),
           np.linspace(-b, b, 10), 
           np.linspace(-b, b, 10))

#Electric Field
def E(q, a1, X, Y, Z):
    E_x = 2*k0*q*(x-a1[0])/(np.sqrt(np.power(np.power(x-a1[0],2) + np.power(y - a1[1],2) + np.power(z - a1[2],2),3)))
    E_y = k0*q*(y-a1[1])/(np.sqrt(np.power(np.power(x-a1[0],2) + np.power(y - a1[1],2) + np.power(z - a1[2],2),3)))
    E_z = k0*q*(z-a1[2])/(np.sqrt(np.power(np.power(x-a1[0],2) + np.power(y - a1[1],2) + np.power(z - a1[2],2),3)))
    return E_x,E_y,E_z 

def V(q, a2, X, Y, Z):
    V_x = 2*k0*q*(x-a1[0])/(np.sqrt(np.power(np.power(x-a1[0],2) + np.power(y - a1[1],2) + np.power(z - a1[2],2),2)))
    V_y = k0*q*(y-a1[1])/(np.sqrt(np.power(np.power(x-a1[0],2) + np.power(y - a1[1],2) + np.power(z - a1[2],2),2)))
    V_z = k0*q*(z-a1[2])/(np.sqrt(np.power(np.power(x-a1[0],2) + np.power(y - a1[1],2) + np.power(z - a1[2],2),2)))
    return V_x,V_y,V_z

def B1(I0, a3, X, Y, Z):
    B_x = (mu_0*I0*(y - a3[1]))/(np.sqrt(np.power(np.power(x-a3[0],2) + np.power(y - a3[1],2),3)))
    B_y = -(mu_0*I0*(x - a3[0]))/(np.sqrt(np.power(np.power(x-a3[0],2) + np.power(y - a3[1],2),3)))
    B_z = np.zeros_like(np.power(B_x,2)+np.power(B_y,2))
    return B_x,B_y,B_z

# calculate vector function of electric field
Ex1, Ey1, Ez1 = E(Q, [0, 0, 0], x, y, z)
Ex2, Ey2, Ez2 = E(-Q, [l, 0, 0], x, y, z)
Ex = Ex1 + Ex2
Ey = Ey1 + Ey2
Ez = Ez1 + Ez2

#calculate vector function of magnetic field perpendicular to the electric field
Bx1_1, By1_1, Bz1_1 = B1(I, [0,0], x, y, z)
Bx2_1, By2_1, Bz2_1 = B1(I, [0,0], x, y, z)
Bx1 = Bx1_1 + Bx2_1
By1 = By1_1 + By2_1
Bz1 = Bz1_1 + Bz2_1

B_max = 1e10
B = np.sqrt(np.power(Bx1,2)+np.power(By1,2)+np.power(Bz1,2))
k1 = np.where(B.flat[:]>B_max)
Ex.flat[k1] = np.nan
Ey.flat[k1] = np.nan
Ez.flat[k1] = np.nan


# remove vector with length larger than E_max
E_max = 1e10
E = np.sqrt(np.power(Ex,2)+np.power(Ey,2)+np.power(Ez,2))
k = np.where(E.flat[:]>E_max)
Ex.flat[k] = np.nan
Ey.flat[k] = np.nan
Ez.flat[k] = np.nan

# Plot the plasma carrying the current
plasma_lenght = np.linspace(0, l, 50)
plasma_rx = np.zeros_like(plasma_lenght)
plasma_ry = np.zeros_like(plasma_lenght)
ax.plot(plasma_lenght, plasma_rx, plasma_ry, lw=3, color='red')

# Create the quiver plot of the elctric field
#ax.quiver(x, y, z, Ex, Ey, Ez, length = 1e-3, color = "blue", normalize=True, cmap = plt.cm.jet)

# Create the quiver plot of the magnetic field
ax.quiver(x, y, z, Bx1, By1, Bz1, length = 1e-3, color = "red", normalize=True, cmap = plt.cm.jet)

ax.set_xlabel("E(lenght axis(m))")
ax.set_ylabel("E(x axis(m))")
ax.set_zlabel("E(y axis(m))")
ax.set_title("Electric Field of the Plasma System")
ax.set_xlim(0, l)
ax.set_ylim(-b, b)
ax.set_zlim(-b, b)


plt.show()
