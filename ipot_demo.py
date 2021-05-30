
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 21:54:13 2017

@author: yujia
"""


import ipot
import numpy as np
import ot
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

   
n = 100

###########data#############
# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions for input
a1 = 0.5*ot.datasets.get_1D_gauss(n, m=70, s=9)+0.5*ot.datasets.get_1D_gauss(n, m=35, s=9)  # m= mean, s= std
a2 = 0.4*ot.datasets.get_1D_gauss(n, m=60, s=8)+0.6*ot.datasets.get_1D_gauss(n, m=40, s=6)

print('This is the two input margins.' )
plt.plot(x, a1,'o-',color = 'orange',markersize=3)
plt.plot(x, a2,'o-',markersize=3)
plt.title('Margins', fontsize=20)
plt.tight_layout()
plt.show()

# loss matrix + normalization
C = ot.utils.dist0(n)
C /= C.max()


# wasserstein
T=ot.emd(a1,a2,C) #This is a good function to obtain LP result. It usually fails when n>5000
ground_truth = np.sum(T*C)


# settings
num_proximal = 2000
beta_list = [0.01,0.1,1]
L=1
use_path = True


loss_list = []
P_list = []

for beta in beta_list:

    P,loss = ipot.ipot_WD(a1,a2,C,beta=beta,max_iter=num_proximal,L=L,use_path = use_path)
    loss_list.append(np.asarray(loss))
    P_list.append(P)
    

#%% This part is for figures
## This is the transportatiopn plan
    
print("This is the transportation plan under different beta.")
    
import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('Reds')
new_cmap = truncate_colormap(cmap, 0., 0.8)
    
f, axarr = plt.subplots(1,len(beta_list),figsize=(9,3))
for i,beta in enumerate(beta_list):
        
        axarr[i].imshow(P_list[i],cmap=new_cmap)
        axarr[i].imshow(T,cmap=plt.get_cmap('binary'),alpha=0.7)
        axarr[i].yaxis.set_ticks([])
        axarr[i].xaxis.set_ticks([])
        axarr[i].set_title(r'$\beta$ = '+str(beta), fontsize=20)
        
plt.show()

## This is for the convergence
print('This is to show the convergence behavoir.')
from cycler import cycler
plt.rc('axes', prop_cycle=(cycler('color', ['darkred', 'r', 'salmon']) ))

for i,opt1 in enumerate(beta_list):
    plt.semilogy(np.asarray(loss_list[i])-ground_truth,label=r'IPOT $\beta$ = '+str(beta_list[i]))

plt.ylabel('$|W-W_{LP}|$',fontsize=20)
plt.xlabel('# iteration',fontsize=20)
plt.xlim([-100,num_proximal+100])

plt.legend(fontsize=15,loc=1, bbox_to_anchor=(1.55, 1.))
plt.show()
