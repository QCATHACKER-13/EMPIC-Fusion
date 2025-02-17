import numpy as np
import matplotlib.pyplot as plt

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

# Quantum Particle class
class quant_particle:
    def __init__(self, xi, pi, xf, pf):
        self.xi = xi  # Initial position
        self.pi = pi  # Initial momentum
        self.x = xf #Final position
        self.p = pf #Final momentum
    
    def wavefunction(self):
    	return (1/np.sqrt(np.abs(self.x - self.xi)))*np.exp(-1j*(self.x - self.xi)*(self.p - self.pi))
    
    def electric_potential(self, q, Vab):
    	return (q*Vab*(self.x - self.xi))

# Parameters
num_particles = 100
num_steps = 1000
dt = 0.01

#Position Space Boundary
x_min = -1
x_max = 1

#Initialization of the particle's position
x = np.zeros(num_particles)

#Initialization of the particle's velocity
p = np.zeros(num_particles)

#For normalization value
A = np.sqrt(x_max - x_min)

#Initialize the graph
fig = plt.figure(dpi = 100)
axis = fig.add_subplot(projection = "3d")

# Run the simulation
# Quantum Particle-in-Cell simulation
print("Particle ID         Position        Momentum          Prob")
for step in range(num_steps):
            	#particle's randomization
            	p0 = np.random.uniform(0, me*c0, num_particles)
            	x0 = np.random.uniform(x_min, x_max, num_particles)
            	
            	for particle in range(0, num_particles):
            		#Initializes class
            		quant = quant_particle(x[particle], p[particle], x0[particle], p0[particle])
            		# Update momentum based on potential energy gradient
            		p[particle] -= (quant.electric_potential(-e, 2000)*x0[particle]*dt)/h_bar
            		
            		# Update position based on momentum
            		x[particle] += (p0[particle]/ me) *dt
            		prob = np.power(np.abs(quant.wavefunction()), 2)
            		
            		#Print thr numerical result
            		#print(f"particle[{particle}]: {x[particle]}     {p[particle]}")
            		
            		#Boundary condition
            		x[particle] = np.mod(x[particle] - x_min, x_max) + x_min
            		
            		axis.scatter(x, prob, s = 10, marker = 'o', color = "blue")
            		axis.set_xlim(x_min, x_max)
            		axis.set_xlabel("Position")
            		axis.set_ylabel("Probability")
            		axis.set_title("QPIC: Probability of Particle in Position")


plt.show()