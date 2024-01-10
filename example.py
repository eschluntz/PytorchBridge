#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from truss import Truss

# Simple bridge example
node_tensors = np.array([
    (0.0, 0.0),
    (0.0, 1.0), 
    (1.0, 0.0),
    (1.0, 1.0), 
    (2.0, 0.0),
    (2.0, 1.0), 
    (3.0, 0.0),
    (3.0, 1.0), 
    (4.0, 0.0),
    (4.0, 1.0), 
])
edges = np.array((
    (0,2), (2,4), (4,6), (6,8),
    (1,3), (3,5), (5,7), (7,9),
    (0,1), (2,3), (4,5), (6,7), (8,9),
    (0,3), (2,5), (5,6), (7,8),
    (1,2),(3,4),(4,7),(6,9)
))
t = Truss(node_tensors, edges)


t.add_load(2, 0, -5)
t.add_load(4, 0, -5)
t.add_load(6, 0, -5)
t.add_anchor(0)
t.add_anchor(8,x=False)
t.calculate_forces()
# t.draw()
# plt.show()

t.optimize(n_frames=30)

plt.show()