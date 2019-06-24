#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 6 16:29:14 2018

@author: yujia

The code uses tricks in Python package 'POT', thanks to
Flamary, R{\'e}mi and Courty, Nicolas

"""

import numpy as np
    
def ipot_WD(a1,a2,C,beta=2,max_iter=1000,L=1,use_path = True, return_map = True, return_loss = True):
    u"""
    Solve the optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F 

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - C is the (ns,nt) metric cost matrix
    - a and b are source and target weights (sum to 1)

    The algorithm used priximal point method


    Parameters
    ----------
    a1 : np.ndarray (ns,)
        samples weights in the source domain
    a2 : np.ndarray (nt,) or np.ndarray (nt,nbb)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    C : np.ndarray (ns,nt)
        loss matrix
    beta : float, optional
        Step size of poximal point iteration
    max_iter : int, optional
        Max number of iterations
    L : int, optional
        Number of iterations for inner optimization
    use_path : bool, optional
        Whether warm start method is used
    return_map : bool, optional
        Whether the optimal transportation map is returned
    return_loss : bool, optional
        Whether the list of calculated WD is returned


    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    loss : list
        log of loss (Wasserstein distance)

    Examples
    --------

    >>> import ipot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ipot.ipot_WD(a,b,M,beta=1)
    array([[ 1.,  0.],
           [ 0.,  1.]])


    References
    ----------

    [1] Xie Y, Wang X, Wang R, et al. A Fast Proximal Point Method for 
    Wasserstein Distance[J]. arXiv preprint arXiv:1802.04307, 2018.
    
    
    """
    
    n = len(a1)
    v = np.ones([n,])
    u = np.ones([n,])

    P = np.ones((n,n))/n**2

    K=np.exp(-(C/beta))
    if return_loss==True:
        loss = []
    for outer_i in range(max_iter):

        Q = K*P
       
        if use_path == False:
            v = np.ones([n,])
            u = np.ones([n,])
    
        
        for i in range(L):
            u = a1/np.matmul(Q,v)
            v = a2/np.matmul(np.transpose(Q),u)
    
        P = np.expand_dims(u,axis=1)*Q*np.expand_dims(v,axis=0)
        if return_loss==True:
            W = np.sum(P*C) 
            loss.append(W)
            
    if return_loss==True:
        if return_map==True:
            return P, loss
        
        else:
            return loss

    else:
        if return_map==True:
            return P
        
        else:
            return None
        

        
def geometricBar(weights, alldistribT):
    """return the weighted geometric mean of distributions"""
    assert(len(weights) == alldistribT.shape[1])
    return np.exp(np.dot(np.log(alldistribT), weights.T))


def geometricMean(alldistribT):
    """return the  geometric mean of distributions"""
    return np.exp(np.mean(np.log(alldistribT), axis=1))
    
def ipot_barycenter(A, M, beta, weights=None, numItermax=1000):
    """Compute the wasserstein barycenter of distributions A
     The function solves the following optimization problem:
    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W(\mathbf{a},\mathbf{a}_i)
    where :
    - :math:`W(\cdot,\cdot)` is the Wasserstein distance 
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
    - :math:`\mathbf{M}` is the cost matrix for OT
    
    The algorithm absorbs many tricks in Python package "POT".
    
    Parameters
    ----------
    A : np.ndarray (d,n)
        n training distributions of size d
    M : np.ndarray (d,d)
        loss matrix   for OT
    beta : float
        Step size of rpoximal point iteration
    numItermax : int, optional
        Max number of iterations

    Returns
    -------
    a : (d,) ndarray
        Wasserstein barycenter

    References
    ----------
    [1] Xie Y, Wang X, Wang R, et al. A Fast Proximal Point Method for 
    Wasserstein Distance[J]. arXiv preprint arXiv:1802.04307, 2018.
    [2] Flamary, R{\'e}mi and Courty, Nicolas, POT Python Optimal Transport library, 2017
    """

    if weights is None:
        weights = np.ones(A.shape[1]) / A.shape[1]
    else:
        assert(len(weights) == A.shape[1])

        
    n,k = np.shape(A)
    
    cpt = 0
    
    # M = M/np.median(M) # suggested by G. Peyre
    K = np.exp(-M / beta)
    
    Pi = np.ones((n,n,k))/(n*n)
    
    K = np.expand_dims(K,axis=2)
    
    Q = K*Pi
    
    v = np.divide(A, np.sum(Q,axis=0))
    UKv = np.sum(Q*np.expand_dims(v,axis=0),axis=1)
    u = (geometricMean(UKv) / UKv.T).T
    
    while (cpt < numItermax):
        cpt = cpt + 1
        
        Q = K*Pi
        for i in range(1):
            
            u = (u.T * geometricBar(weights, UKv)).T / UKv #for numerical stable
            v = np.divide(A, np.sum(Q*np.expand_dims(u,axis=1),axis=0))
            Pi = np.expand_dims(u,axis=1)*Q*np.expand_dims(v,axis=0)
            Pi /= np.sum(Pi,axis=(0,1)) #for numerical stable
            UKv = u * np.sum(Q*np.expand_dims(v,axis=0),axis=1)

    return geometricBar(weights, UKv)