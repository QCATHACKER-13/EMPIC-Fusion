#Electromagnetic Particle-In-Cell Method
#Developer and Researcher: Christopher Emmanuelle J. Visperas

import matplotlib.pyplot as plt
import numpy as np
import cfield as em
from pointparticle import particle
from geo import Plane
from geo import Solid


# constant
epsilon_0 = 8.854187817e-12 #permittivity of free space
k0 = 1/(4*np.pi*epsilon_0)
e = 1.60217663e-19 # elementary charge
h_bar = 1.054571628e-34 #reduced plank's constant (J*s)
h = 6.62606896e-34 # plank's constant (J*s)
mu_0 = 1.256637061e-6 # permeability of free space
mp = 1.672621637e-27 # kg, mass of a proton
mn = 1.674927211e-27 # kg, mass of a neutron
me = 9.10938215e-31 # kg, mass of an electron
c0 = 299792458 # m/s, speed of light
alpha = k0*(np.power(e,2)/(h_bar*c0)) #fine tune constant
r0_mim = 1.2e-15*np.cbrt(2) # minimum nuclear radius parameter
r0_max = np.power(2*np.power(e, 2)*np.sqrt(k0)*(np.sqrt(mp)/h_bar), 2)
mH = 1.6735575e-27 # kg, mass of hydrogen atom
density_H = 0.0899 # kg/m^3, density of hydrogen
Up = 2*((me*mp)/(me + mp))*np.power(c0,2) #potential of proton

# Physical Parameters
a = 0.5e-2 #center gap distance between two elctrodes
b = 0.15e-2 #radii of the electrode
d = 2.54e-2 #diameter of the cylinder glass 1 inch
l1 = 3.175e-2 #lenght of the cylinder glass 1.25 inch
L_plasma = 1e-2 #lenght of the plasma channel
Q = 3.3e-6 #1.32e-5 for 150000 #3.0e-6 for 75000 # Coulumb length=0.2, 
I = np.pi*Q*np.power(0.5e-2,2)
n = 47875.72074 #total of turns = no. of turns in lenght * no. of turns in radii
R = np.sqrt(np.power(0.0802,2) - np.power(0.038,2))#m, final radii of the electromagnet and initial radii of the electromagnet
l2 = 0.056 #m, lenght of the electromagnet
Vab = 150000
volume = np.pi*np.power(d/2,2)*l1
ve = np.sqrt((2*Vab*e)/me) #for electron velocity
vp = np.sqrt((2*Vab*e)/mp) #for proton velocity

#

#number of coordinates (x, y, z)
Nx, Ny, Nz = 100, 100, 100

#Coordinate is used
solid = Solid([-b, -b, a], [b, b, l1 - a], [Nx, Ny, Nz])
x, y, z = solid.cartesian()

print(x, y, z)

#number of particles 
num_particles = [1, 5]

#initial position
re = np.random.uniform(0, b, num_particles[0])
theta_e = np.random.uniform(0, 2*np.pi, num_particles[0])
ze = np.random.uniform(a, l1 - a, num_particles[0])

#initial velocity
ve_x = np.random.uniform(-c0, c0, num_particles[0])

# Create the figure and axes
fig = plt.figure(dpi = 100)
axis = fig.add_subplot(projection = "3d")

#time step
dt = 1e-3

#time initial
ta = 0.0

#time total
tb = 60

no_step = 0.0
total_steps = tb/dt


#step numbers

while no_step < (total_steps + 1):

    no_step+= 1

