#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:36:41 2018

@author: yujia
"""

import numpy as np
import ot
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
import ipot

    
batchsize = 100

###########data#############
n = batchsize  # n bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a1 = ot.datasets.get_1D_gauss(n, m=20, s=5)+ot.datasets.get_1D_gauss(n, m=40, s=5)  # m= mean, s= std
a2 = ot.datasets.get_1D_gauss(n, m=60, s=8)

# creating matrix A containing all distributions
A = np.vstack((a1, a2)).T
n_distributions = A.shape[1]

# loss matrix + normalization
M = ot.utils.dist0(n)
M /= M.max()


# barycenter computation

alpha = 0.5  # 0<=alpha<=1
weights = np.array([1 - alpha, alpha])

# l2bary
bary_l2 = A.dot(weights)

# sinkhorn
reg = 0.001
bary_wass1 = ot.barycenter(A, M, reg, weights,numItermax=400)

#ipot

reg = 0.01
bary_wass2 = ipot.ipot_barycenter(A, M, reg, weights,numItermax=100)

plt.figure(2)
plt.clf()
plt.subplot(2, 1, 1)
for i in range(n_distributions):
    plt.plot(x, A[:, i])
plt.title('Distributions')

plt.subplot(2, 1, 2)
plt.plot(x, bary_l2, 'r', label='l2')
plt.plot(x, bary_wass1, 'g', label='Sinkhorn')
plt.plot(x, bary_wass2, 'b', label='Proximal')
plt.legend(loc=1, bbox_to_anchor=(1.45, 1.1))
plt.title('Barycenters')
plt.tight_layout()
plt.show()