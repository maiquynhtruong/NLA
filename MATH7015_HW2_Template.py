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
from scipy.linalg import lu, solve_triangular, lu_factor, lu_solve

# Problem #2
def MultiLinearSolve(A, bs):
    xs = [] # List of solutions x_i such that A x_i = b_i for i = 0 to M-1
    for k in range(len(bs[0])): # len(b[0]) to get the number of columns
        xs.append(np.linalg.solve(A, bs[:,k]))

    return np.stack(xs).T # put the results into an array and transpose them

# Problem #3
################################### First implementation ###################################
def MultiLUsolve(A, bs):
    LU decomposition of A. p is permutation matrix
    p, l, u = lu(A)
    ys, xs = [], []

    bs = MultiLinearSolve(p,bs) # Instead of b, we need  p-1 b and we can reuse MultiLinearSolve for that

    # The original system is LUx = b. Now find y such that Ly = b
    for k in range(len(bs[0])):
        ys.append(solve_triangular(l, bs[:,k], lower=True))
    ys = np.stack(ys).T

    # Then find x such that Ux = y
    for k in range(len(ys[0])):
        xs.append(solve_triangular(u, ys[:,k]))
    xs = np.stack(xs).T

    return xs

################################### Second implementation ###################################
#
# def MultiLUsolve(A, bs):
#     xs = []
#     lu, piv = lu_factor(A) # get LU factorizatin of A
#
#     for k in range(len(bs[0])):
#         xs.append(lu_solve((lu, piv), bs[:,k])) # solve for x with L and U
#     xs = np.stack(xs).T
#
#     return xs
