#Electromagnetic Particle-In-Cell Method
#For Computational Physics Reseach Purposes
#Primarily study and focus on plasma physics and nuclear fusion research
#Start from Non - Relativistic and Relativistic Lorentz Force Mechanics
#Then progressively develop in much further theory application like Lattice Field Theory,
#Quantum Mechanics, Relativistic Quantum Mechanics, QuantumFT, QuantumED, QuantumCD and etc.
#Developer and Researcher: Christopher Emmanuelle J. Visperas

from geo import Plane, Solid
import numpy as np
import sys

class Sgrid:
    def __init__(self, num_cells : int, a : float, b : float):
        
        self.num_cells = num_cells #number of cells

        #Domain of x, y, z axis from the finite difference

        #From lower limit of the axis
        self.xa = a[0]
        self.ya = a[1]
        self.za = a[2]

        #From upper limit of the axis
        self.xb = b[0]
        self.yb = b[1]
        self.zb = b[2]

    def space(self, dimension, coordinates):
        match dimension:
            case 2:
                plane = Plane(self.num_cells, [self.xa, self.ya], [self.xb, self.yb])

                match coordinates:
                    case "cartesian":
                        return plane.cartesian()
                    
                    case "polar":
                        return plane.polar()
                    
                    case _:
                        print("type your string identify 2D coordinate system: 'cartesian' or 'polar' ")
                        return sys.exit()
            
            case 3: 
                solid = Solid(self.num_cells, [self.xa, self.ya, self.za], [self.xb, self.yb, self.zb])

                match coordinates:
                    
                    case "cartesian":
                        return solid.cartesian()
                    
                    case "cylindrical":
                        return solid.cylindrical()
                    
                    case "spherical":
                        return solid.spherical()
                    
                    case _:
                        print("type your string identify 3D coordinate system: 'cartesian' or 'polar' ")
                        return sys.exit()
            
            case _:
                print("type your input integer to identify the dimension")
                return sys.exit()