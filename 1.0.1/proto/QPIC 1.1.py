import matplotlib.pyplot as plt
import numpy as np
from emfield import Field as field
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

# Parameters
a = 0.5e-2 #center gap distance between two elctrodes
b = 0.15e-2 #radii of the electrode
d = 2.54e-2 #diameter of the cylinder glass 1 inch
l1 = 3.175e-2 #lenght of the cylinder glass 1.25 inch
L_plasma = 1e-2 #lenght of the plasma channel
Q = 3.3e-6#1.32e-5 for 150000 #3.0e-6 for 75000 # Coulumb length=0.2, 
I = np.pi*Q*np.power(0.5e-2,2)
I_external = 3.0
n = 47875.72074 #total of turns = no. of turns in lenght * no. of turns in radii
R = np.sqrt(np.power(0.0802,2) - np.power(0.038,2))#m, final radii of the electromagnet and initial radii of the electromagnet
l2 = 0.056 #m, lenght of the electromagnet
Vab = 150000
volume = np.pi*np.power(d/2,2)*l1
ve = np.sqrt((2*Vab*e)/me) #for electron velocity
vp = np.sqrt((2*Vab*e)/mp) #for proton velocity
ds = 10
dt = 1e-15# time step
delay = 1e-6
t_max = 1e-11          # Maximum simulation time
total_steps = 2684
listing_no = 0

# Quantum Particle class
class quant_particle:
    def __init__(self, x, p):
        self.x = x  # Position
        self.p = p  # Momentum
    
    def wavefunction(self):
        psi = np.exp((-1j*self.x*self.p)/h_bar)
        return psi

# Parameters
num_particles = 100
num_steps = 1000
dt = 0.01
N = 100 #Space Resolution

#Position Space Boundary
x_min, x_max = -1, 1

# Slope of the space
delta_x = (x_max - x_min)/N

#Initialization of the particle's position
x_particle = np.random.uniform(x_min, x_max, num_particles) 

#Initialization of the particle's velocity
p_particle = np.random.uniform(-me*ve, me*ve, num_particles)

#Declaring the class for space
plane = Plane([x_min, 0], [x_max, 0], [100, 100])

#Declare the lattice space parameter
x, y = plane.cartesian()

#Intializing the field in a space lattice
field_a = field([x_min, 0, 0], [x + delta_x, 0, 0])
field_b = field([x + delta_x, 0, 0], [x_max, 0, 0])
E1_x, E1_y, E1_z = field_a.E(Q)
E2_x, E2_y, E2_z = field_b.E(-Q)
Ex, Ey, Ez = E1_x + E2_x, E1_y + E2_y, E1_z + E2_z 

#Initialize the graph
plt.grid()

# Run the simulation
# Quantum Particle-in-Cell simulation
print("Particle ID         Position        Momentum          Prob")
for step in range(num_steps):
        for particle in range(1, num_particles):
                #Initializes class for field
                field_sa = field([x_min, 0, 0], [x_particle[particle], 0, 0])
                field_sb = field([x_particle[particle], 0, 0], [x_max, 0, 0])
                
                #Initializes class for quantum part
                quant = quant_particle(x_particle[particle], p_particle[particle])
                
                #Probability of Particle's Position
                prob = np.abs(np.real(quant.wavefunction())*np.imag(quant.wavefunction()))

                #Cell Index
                cell_index = int(np.abs(x_particle[particle])/delta_x)
                # Interpolating of the Electric Potential
                E1 = field_sa.E(Q) + field_sb.E(-Q)
                E = E1 + ((x_particle[particle] - x[cell_index])/(2*delta_x))*(Ex[cell_index] - E1)

                #Final momentum based on potential energy gradient
                p_particle[particle] += (-e*(Ex[int(np.abs(x_particle[particle] + delta_x))] + Ex[int(np.abs(x_particle[particle] - delta_x))]))*dt
                
                # Final position based on momentum
                x_particle[particle] += (p_particle[particle]/me)*dt
                
                #Print thr numerical result
                #print(f"particle[{particle}]: {x_particle[particle]}     {p_particle[particle]}     {prob}")
                
                #Boundary condition
                x_particle[particle] = np.mod(x_particle[particle] - x_min, x_max - x_particle[particle])
                
                plt.plot(x_particle[particle], prob, "ro")
                plt.xlim(x_min, x_max)
                plt.ylim(0, 1)
                plt.xlabel("Position")
                plt.ylabel("Probability")
                plt.title("QPIC: Probability of Particle in Position")
                plt.pause(0.1)


plt.show()