import numpy as np


a = 0.5e-2
l1 = 3.175e-2

x = np.random.uniform(a, l1 - a, 1)

while not(x == (l1 - a)):
    print(x)
    x += (((l1 - a) - a)/100)
    
