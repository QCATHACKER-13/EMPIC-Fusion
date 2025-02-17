from geo import Plane, Solid
from emfield import Field
import numpy as np
import sys

class Grid:
    def __init__(self, num_cells, a, b):
        
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

    def Lattice_Field(self, dimension, coordinates, Q, N, I):
        match dimension:
            case 2:
                plane = Plane(self.num_cells, [self.xa, self.ya, self.za], [self.xb, self.yb, self.zb])

                match coordinates:
                    case "cartesian":
                        x, y = plane.cartesian()
                        field_pointa = Field([self.xa, self.ya, 0], [x, y, 0])
                        field_pointb = Field([x, y, 0], [self.xb, self.yb, 0])

                        E_field = field_pointa.E(Q) + field_pointb.E(-Q)
                        B_field = field_pointa.B_induced(N, I) + field_pointb.B_induced(N, -I)
                        
                        return E_field, B_field
                    
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