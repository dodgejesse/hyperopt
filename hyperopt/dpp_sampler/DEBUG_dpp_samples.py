import matlab_wrapper
print("about to start MatlabSession()..."),
mtb=matlab_wrapper.MatlabSession()
print("done!")
import GPy
import matplotlib.pyplot as plt
from GPyOpt.util.general import multigrid

import pdb; pdb.set_trace()
import os
cwd = os.getcwd()
print("current working directory: {}".format(cwd))

from dpp import *

# Genetate grid
Ngrid = 50
bounds = [(-2,2),(-2,2)]
X = multigrid(bounds, Ngrid)  

# Define kernel and kernel matrix
kernel = GPy.kern.RBF(len(bounds), variance=1, lengthscale=0.5) 
L = kernel.K(X)

# Number of points of each DPP sample
k = 50

# Putative inputs
selected = [25,900, 1655,2125]

# Samples and plot from original and conditional with standard DPPS
print("about to call sample_dpp(L,k)..."),
sample          = sample_dpp(L,k)
print("done!")
print("about to call sample_conditional_dpp(L,k)..."),
sample_condset  = sample_conditional_dpp(L,selected,k)
print("done!")

plt.subplot(1, 2, 1)
plt.plot(X[sample,0],X[sample,1],'.',)
plt.title('Sample from the DPP')
plt.subplot(1, 2, 2)
plt.plot(X[selected,0],X[selected,1],'k.',markersize=20)
plt.plot(X[sample_condset,0],X[sample_condset,1],'.',)
plt.title('Conditional sample from the DPP')
plt.show()
