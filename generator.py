import math
import random 

random.seed(1)

num_points = 20000
dimension = 60
K = 50
max_iter = 500
has_name = 0

# Generate Gaussian Distribution
with open('datasets/dataset3.txt', 'w') as f:
    params = '{} {} {} {} {}\n'.format(str(num_points), str(dimension), str(K), str(max_iter), str(has_name))
    lines = [params]
    for i in range(num_points):
        line = [str(random.gauss(0, 10)) for j in range(dimension)]
        line = ' '.join(line)
        lines.append(f'{line}\n')
    f.writelines(lines)

""" Generate Spherical Gaussian Distribution 

import numpy as np
def spherical_to_cartesian(vec):                                                  
    (r, u, v) = vec                                                                     
    x = float(r * math.sin(u) * math.cos(v))
    y = float(r * math.sin(u) * math.sin(v))                                                   
    z = float(r * math.cos(u))
    return [x, y, z]  

U = np.random.random(num_points)
V = np.random.random(num_points)
radii = np.random.normal(10, 3, num_points)
points = np.array([spherical_to_cartesian([r, 2 * np.pi * u, np.arccos(2*v - 1)]) for r, u, v in zip(radii, U, V)])

with open('datasets/dataset3.txt', 'w') as f:
    params = '{} {} {} {} {}\n'.format(str(num_points), str(dimension), str(K), str(max_iter), str(has_name))
    lines = [params]
    for point in points:
        line = ' '.join([str(v) for v in point])
        line += '\n'
        lines.append(line)
    f.writelines(lines)
"""


