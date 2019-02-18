import sys
import numpy as np
import scipy as sp
info = sys.version_info

print("You are running Python v{}.{}.{} with NumPy v{} and SciPy v{}".format(info[0], info[1], info[2], np.__version__, sp.__version__))