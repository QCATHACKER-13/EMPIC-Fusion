import numpy as np

class Grid:
    def __init__(self, num_cells, domain_size):
        self.num_cells = num_cells
        self.domain_size = domain_size
        self.cell_size = domain_size / num_cells
        self.charge_density = np.zeros(num_cells)
        self.potential = np.zeros(num_cells)
        self.electric_field = np.zeros(num_cells)
