#Electromagnetic Particle-In-Cell Method
#For Computational Physics Reseach Purposes
#Primarily study and focus on plasma physics and nuclear fusion research
#Start from Non - Relativistic and Relativistic Lorentz Force Mechanics
#Then progressively develop in much further theory application like Lattice Field Theory,
#Quantum Mechanics, Relativistic Quantum Mechanics, QuantumFT, QuantumED, QuantumCD and etc.
#Developer and Researcher: Christopher Emmanuelle J. Visperas

from grid_space import Sgrid
from grid_field import Fgrid
import numpy as np

#parameters
Q = 3.3E-6
I = 3.5E-3
N = 250

a = [-1, -1, 1]
b = [1, 1, 2]


#Dimensionality
dimension = 3
coordinate_sys = "cartesian"

x, y, z = Sgrid(1000, a, b).space(dimension, coordinate_sys)


print(x, y, z)

#lattice field intialize
#Ea = Fgrid(a, x).Efield(dimension, coordinate_sys, Q)
#Eb = Fgrid(x, b).Efield(dimension, coordinate_sys, -Q)

#E = Ea + Eb

#print(Ea)
#print(E)











