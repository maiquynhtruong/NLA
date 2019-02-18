# Tests student's code for Assignment #2
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
#  How to run the test script:
#    1) Ensure that your source code and this test script are in the current
#           working directory
#
#    2) Import the test script using the command 
#               import MATH7015_HW2_Test
#
#    3) Call the test script using the command
#               MATH7015_HW2_Test.HW2_Test(FILENAME)
#           replacing FILENAME with the actual filename of the code you want to
#           test using quotes and without the py extension. 
#			Example: MATH7015_HW2_Test.HW2_Test("TestCode") will test the code 
#                      in the file TestCode.py
#    
#    4) The test script will output "Tests Passed" if the code passes all the 
#           tests otherwise it will output which tests have failed
#
#    5) Depending on how fast your computer is, it might take awhile for the 
#           tests to finish
#   

import numpy as np

def HW2_Test(filename):
    # Dynamically load the module to test
    import importlib

    
    mod = importlib.import_module(filename)    
    MultiLinearSolve = getattr(mod, "MultiLinearSolve")
    MultiLUsolve = getattr(mod, "MultiLUsolve")
        
    from sys import stdout
    import time
    
    flag = True     # Flag indicating pass/fail
    


    # Test Problems #2 and #3
    tol = 1e-8
    vN = np.array([2, 5, 10, 100, 1000, 2000, 4000])
    M = 100

    nTrials = len(vN)

    for kk in range(nTrials):
        N = vN[kk]
        print("Testing for N = {}".format(N))

        A = 2*np.identity(N) + np.random.rand(N, N)
        xs = np.random.rand(N, M)

        bs = np.empty_like(xs)
        for jj in range(M):            
            bs[:, jj] = np.dot(A,xs[:, jj])
        
        memUsage = A.nbytes + bs.nbytes + xs.nbytes
        print("Memory Usage: {:6.3f} MB".format(memUsage/1024/1024))

        startTime = time.time()
        xGauss = MultiLinearSolve(A, bs)
        endTime = time.time()
        print("Gaussian Elimination Time: {:6.3f} seconds".format(endTime - startTime))

        startTime = time.time()
        xLU = MultiLUsolve(A, bs)
        endTime = time.time()
        print("LU Factorization Time: {:6.3f} seconds".format(endTime - startTime))

        gaussError = np.max(xs - xGauss)
        luError = np.max(xs - xLU)

        if gaussError > tol:
            flag = False
            print("Test failed for MultiLinearSolve: Error = {:6.3e}".format(gaussError))
        
        if luError > tol:
            flag = False
            print("Test failed for MultiLUSolve: Error = {:6.3e}".format(luError))
    print("Problem #2 and #3 Tests Finished."); stdout.flush()
    
    
    
    if flag:
        print("Tests passed!")
    else:
        print("Tests failed!")
        
    return flag
        