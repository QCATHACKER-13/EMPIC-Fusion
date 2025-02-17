import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#constant
epsilon_0 = 8.854187817e-12
k0 = 1/(4*np.pi*epsilon_0)
e = 1.60217663e-19
mu_0 = 1.256637061e-6
mp = 1.672621637e-27 #kg, mass of a proton
mn = 1.674927211e-27 #kg, mass of a neutron
me = 9.10938215e-31 #kg, mass of an electron
c0 = 299792458 #m/s, speed of light
h = 6.62607015e-34 #J*s, plank's constant
h_bar = h/(2*np.pi) #J*s, reduced plank's constant

#Parameters
a = 0.5e-2 #center gap distance between two elctrodes
b = 0.15e-2 #radii of the electrode
d = 1.0e-2 #diameter of the cylinder glass
l1 = 2.50e-2 #lenght of the cylinder glass
Q = 3.0e-11 #aCoulumb length=0.2, 
I = np.pi*Q*np.power(0.5e-2,2)
n = 47875.72074 #total of turns = no. of turns in lenght * no. of turns in radii
R = np.sqrt(np.power(0.0802,2) - np.power(0.038,2))#m, final radii of the electromagnet and initial radii of the electromagnet
l2 = 0.056 #m, lenght of the electromagnet
Vab = 150000
ve = np.sqrt((2*Vab*e)/me) #for electron velocity
vp = np.sqrt((2*Vab*e)/mp) #for proton velocity
gamma = [1/np.sqrt(1 - np.power(ve/c0,2)),1/np.sqrt(1 - np.power(vp/c0,2))]
ds = 10

# Create the figure and axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d') 


# Make the grid
x, y, z = np.meshgrid(np.linspace(-b, b, ds), np.linspace(-b, b, ds), np.linspace(0, l1, ds))

x1, y1, z1 = np.meshgrid(np.linspace(-R, R, ds),np.linspace(-R, R, ds),np.linspace(0, l2, ds))


#Electric Field
def E(q, a1, X, Y, Z):
    r = np.sqrt(np.power(X - a1[0],2) + np.power(Y - a1[1],2) + np.power(Z - a1[2],2))
    E_x = k0*q*(X - a1[0])/np.power(r,3) 
    E_y = k0*q*(Y - a1[1])/np.power(r,3)
    E_z = 2*k0*q*(Z - a1[2])/np.power(r,3)
    return E_x,E_y,E_z 

def V(q, a2, X, Y, Z):
    V_x = 2*k0*q*(X-a2[0])/(np.sqrt(np.power(np.power(x-a2[0],2) + np.power(y - a2[1],2) + np.power(z - a2[2],2),2)))
    V_y = k0*q*(Y-a2[1])/(np.sqrt(np.power(np.power(x-a2[0],2) + np.power(y - a2[1],2) + np.power(z - a2[2],2),2)))
    V_z = k0*q*(Z-a2[2])/(np.sqrt(np.power(np.power(x-a2[0],2) + np.power(y - a2[1],2) + np.power(z - a2[2],2),2)))
    return V_x,V_y,V_z

def B1(I0, a3, X, Y, Z):
    r = np.sqrt(np.power(X-a3[0],2) + np.power(Y - a3[1],2) + np.power(Z - a3[2],2))
    B_x1 = (mu_0*I0*(Y - a3[1]))/(2*np.pi*np.power(r,3))
    B_y1 = -(mu_0*I0*(X - a3[0]))/(2*np.pi*np.power(r,3))
    B_z1 = np.zeros_like(r)
    return B_x1,B_y1,B_z1

def B2(I0, N, a4, X, Y, Z):
    r = np.sqrt(4*np.power(X-a4[0],2) + 4*np.power(Y - a4[1],2) + np.power(Z - a4[2],2))
    B_x2 = (mu_0*I0*N*(X - a4[0]))/(2*np.pi*np.power(r,3))
    B_y2 = (mu_0*I0*N*(Y - a4[1]))/(2*np.pi*np.power(r,3))
    B_z2 = (mu_0*I0*N*(Z - a4[2]))/(2*np.pi*np.power(r,3))
    return B_x2,B_y2,B_z2

def A1(I0, a4, X, Y, Z):
    r = np.sqrt(np.power(X-a4[0],2) + np.power(Y - a4[1],2) + np.power(Z - a4[2],2))
    A1_x = -(mu_0*I0*(X - a4[0])*(Z - a4[2]))/(2*np.pi*r*(np.power(X - a4[0],2) + np.power(Y - a4[1],2)))
    A1_y = -(mu_0*I0*(Y - a4[1])*(Z - a4[2]))/(2*np.pi*r*(np.power(X - a4[0],2) + np.power(Y - a4[1],2)))
    A1_z = ((mu_0*I0)/(2*np.pi*r))*((np.power(Y-a4[1],2)/(np.power(X - a4[0],2) + np.power(Z - a4[2],2))) + (np.power(X - a4[0],2)/(np.power(Y - a4[1],2) + np.power(Z - a4[2],2))))
    return A1_x, A1_y, A1_z

def A2(I0, N, a5, X, Y, Z):
    r = np.sqrt(np.power(X-a5[0],2) + np.power(Y - a5[1],2) + np.power(Z - a5[2],2))
    A2_x = ((mu_0*I0)/(2*np.pi*r))*((((Y - a5[1])*(Z - a5[2]))/(np.power(X-a5[0],2) + np.power(Y - a5[1],2))) - (((Y - a5[1])*(Z - a5[2]))/(np.power(X - a5[0],2) + np.power(Z - a5[2],2))))
    A2_y = ((mu_0*I0)/(2*np.pi*r))*((((X - a5[0])*(Z - a5[2]))/(np.power(Y-a5[1],2) + np.power(Z - a5[2],2))) - (((X - a5[0])*(Z - a5[2]))/(np.power(X - a5[0],2) + np.power(Y - a5[1],2))))
    A2_z = ((mu_0*I0)/(2*np.pi*r))*((((X - a5[0])*(Y - a5[1]))/(np.power(X - a5[0],2) + np.power(Z - a5[2],2))) + (((X - a5[0])*(Y - a5[1]))/(np.power(Y - a5[1],2) + np.power(Z - a5[2],2))))
    return A2_x, A2_y, A2_z

# calculate vector function of electric field
Ex1, Ey1, Ez1 = E(Q, [0, 0, 0], x, y, z)
Ex2, Ey2, Ez2 = E(-Q, [0, 0, l1], x, y, z)
Ex = Ex1 + Ex2
Ey = Ey1 + Ey2
Ez = Ez1 + Ez2

#calculate vector function of magnetic field perpendicular to the electric field
Bx1_1, By1_1, Bz1_1 = B1(I, [0,0,0], x, y, z)
Bx2_1, By2_1, Bz2_1 = B1(I, [0,0,l1], x, y, z)
Bx1 = Bx1_1 + Bx2_1
By1 = By1_1 + By2_1
Bz1 = Bz1_1 + Bz2_1

#calculate vector function of magnetic field of the electromagnet
Bx1_2, By1_2, Bz1_2 = B2(I, n, [0, 0, 0], x1, y1, z1)
Bx2_2, By2_2, Bz2_2 = B2(I, n, [0, 0, l2], x1, y1, z1)
Bx2 = Bx1_2 - Bx2_2
By2 = By1_2 - By2_2
Bz2 = Bz1_2 - Bz2_2

#calculate  magnetic potential vector of the B1
Ax1_1, Ay1_1, Az1_1 = A1(I, [0, 0, 0], x, y, z)
Ax2_1, Ay2_1, Az2_1 = A1(I, [0, 0, l1], x, y, z)
Ax1 = Ax1_1 - Ax2_1
Ay1 = Ay1_1 - Ay2_1
Az1 = Az1_1 - Az2_1

#calculate  magnetic potential vector of the B2
Ax1_2, Ay1_2, Az1_2 = A2(I, n, [0, 0, 0], x, y, z)
Ax2_2, Ay2_2, Az2_2 = A2(I, n, [0, 0, l2], x, y, z)
Ax2 = Ax1_2 - Ax2_2
Ay2 = Ay1_2 - Ay2_2
Az2 = Az1_2 - Az2_2

# remove vector with length larger than E_max
E_max = 5e10
E = np.sqrt(np.power(Ex,2)+np.power(Ey,2)+np.power(Ez,2))
k = np.where(E.flat[:]>E_max)
Ex.flat[k] = np.nan
Ey.flat[k] = np.nan
Ez.flat[k] = np.nan

# remove vector with length larger than B1_max
B1_max = 1e10
B1 = np.sqrt(np.power(Bx1,2)+np.power(By1,2)+np.power(Bz1,2))
k1 = np.where(B1.flat[:]>B1_max)
Bx1.flat[k1] = np.nan
By1.flat[k1] = np.nan
Bz1.flat[k1] = np.nan

#remove vector with lenght larger than B2_max
B2_max = 1e10
B2 = np.sqrt(np.power(Bx2,2)+np.power(By2,2)+np.power(Bz2,2))
k2 = np.where(B2.flat[:]>B2_max)
Bx2.flat[k2] = np.nan
By2.flat[k2] = np.nan
Bz2.flat[k2] = np.nan

# remove vector with length larger than A1_max
A1_max = 1e10
A1 = np.sqrt(np.power(Ax1,2)+np.power(Ay1,2)+np.power(Az1,2))
k3 = np.where(A1.flat[:]>A1_max)
Ax1.flat[k3] = np.nan
Ay1.flat[k3] = np.nan
Az1.flat[k3] = np.nan

# remove vector with length larger than A1_max
A2_max = 1e10
A2 = np.sqrt(np.power(Ax2,2)+np.power(Ay2,2)+np.power(Az2,2))
k4 = np.where(A2.flat[:]>A2_max)
Ax2.flat[k4] = np.nan
Ay2.flat[k4] = np.nan
Az2.flat[k4] = np.nan

# Plot the plasma carrying the current
plasma_lenght = np.linspace(0, l1, 50)
plasma_rx = np.zeros_like(plasma_lenght)
plasma_ry = np.zeros_like(plasma_lenght)
ax.plot(plasma_lenght, plasma_rx, plasma_ry, lw=3, color="blueviolet")

# Create the quiver plot of the elctric field
ax.quiver(z, y, x, Ez, Ey, Ex, length=1e-3, color = "blue", normalize = True, cmap = plt.cm.jet)

# Create the quiver plot of the magnetic field
ax.quiver(z, y, x, Bz1, By1, Bx1, length = 1e-3, color = "red", normalize = True, cmap = plt.cm.jet)

# Create the quiver plot of the magnetic field of the electromagnet
ax.quiver(z, y, x, Bz2, By2, Bx2, length = 1e-3, color = "yellow", normalize = True, cmap = plt.cm.jet)

# Create the quiver plot of the magnetic potential vector of the magnetic field perpendicular to the electric field
ax.quiver(z, y, x, Az1, Ay1, Ax1, length = 1e-3, color = "green", normalize = True, cmap = plt.cm.jet)

# Create the quiver plot of the magnetic potential vector of the electromagnet
ax.quiver(z, y, x, Az2, Ay2, Ax2, length = 1e-3, color = "violet", normalize = True, cmap = plt.cm.jet)

ax.set_xlabel("lenght axis(m)")
ax.set_ylabel("x axis(m)")
ax.set_zlabel("y axis(m)")
ax.set_title("Electric Field of the Plasma System")
ax.set_xlim(0, l1)
ax.set_ylim(-2*b, 2*b)
ax.set_zlim(-2*b, 2*b)

plt.show()
