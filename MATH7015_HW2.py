# Source code for Assignment #2
# Name: ***--> Mai Truong <--**
# Required function definitions:
#
#    Problem #2:
#        Function: MultiLinearSolve(A, bs)
#        Input: A (NxN numpy array)
#               bs (NxM numpy array)
#        Output: x (NxM numpy array)
#
#    Problem #3:
#        Input: A (NxN numpy array)
#               bs (NxM numpy array)
#        Output: x (NxM numpy array)
#

import numpy as np
import scipy.linalg as spla

# Problem #2
def MultiLinearSolve(A, bs):
    xs = []
    for k in range(len(bs[0])):
        xs.append(np.linalg.solve(A, bs[:,k]))

    return np.stack(xs).T

# Problem #3
def MultiLUsolve(A, bs):
    # LU decomposition of A. p is permutation matrix
    p, l, u = spla.lu(A)
    # The original system is LUx = b. Now find y such that Ly = b
    ys = np.stack([spla.solve_triangular(l, b, lower=True) for b in bs])
    # Then find x such that Ux = y
    xs = np.stack([spla.solve_triangular(y, b) for b in bs])
    return xs
