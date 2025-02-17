#Electromagnetic Particle-In-Cell Method
#For Computational Physics Reseach Purposes
#Primarily study and focus on plasma physics and nuclear fusion research
#Start from Non - Relativistic and Relativistic Lorentz Force Mechanics
#Then progressively develop in much further theory application like Lattice Field Theory,
#Quantum Mechanics, Relativistic Quantum Mechanics, QuantumFT, QuantumED, QuantumCD and etc.
#Developer and Researcher: Christopher Emmanuelle J. Visperas

from transformation import Plane, Solid
from emfield import Field
import numpy as np
import sys

class Fgrid:
    def __init__(self, xa : float, xb : float):

        #initial position
        self.xa = xa[0]
        self.ya = xa[1]
        self.za = xa[2]

        #final position
        self.xb = xb[0]
        self.yb = xb[1]
        self.zb = xb[2]

    def Efield(self, dimension, coordinates, Q):

        field = Field([self.xa, self.ya, self.za], [self.xb, self.yb, self.zb])

        match dimension:
            case 2:
                
                match coordinates:
                    case "cartesian":
                        return field.E(Q)
                    
                    case "polar":
                        return Plane([field.E(Q)]).polar()
                    
                    case _:
                        print("type your string identify 2D coordinate system: 'cartesian' or 'polar' ")
                        return sys.exit()
            
            case 3: 

                match coordinates:
                    
                    case "cartesian":
                        return field.E(Q)
                    
                    case "cylindrical":
                        return Solid([field.E(Q)]).cylindrical()
                    
                    case "spherical":
                        return Solid([field.E(Q)]).spherical()
                    
                    case _:
                        print("type your string identify 3D coordinate system: 'cartesian' or 'polar' ")
                        return sys.exit()
            
            case _:
                print("type your input integer to identify the dimension")
                return sys.exit()
    
    def Bfield(self, dimension, coordinates, N, I):

        field = Field([self.xa, self.ya, self.za], [self.xb, self.yb, self.zb])

        match dimension:
            case 2:
                
                match coordinates:
                    case "cartesian":
                        return field.B_induced(N, I)
                    
                    case "polar":
                        return Plane([field.B_induced(N, I)]).polar()
                    
                    case _:
                        print("type your string identify 2D coordinate system: 'cartesian' or 'polar' ")
                        return sys.exit()
            
            case 3: 

                match coordinates:
                    
                    case "cartesian":
                        return field.B_induced(N, I)
                    
                    case "cylindrical":
                        return Solid([field.B_induced(N, I)]).cylindrical()
                    
                    case "spherical":
                        return Solid([field.B_induced(N, I)]).spherical()
                    
                    case _:
                        print("type your string identify 3D coordinate system: 'cartesian' or 'polar' ")
                        return sys.exit()
            
            case _:
                print("type your input integer to identify the dimension")
                return sys.exit()