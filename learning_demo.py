#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:42:20 2018

@author: yujia

The noise and inout data are both of uniform distribution

The generator is 1-layer NN

The gradient update is using envelop theorem

"""

import ipot
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

#Settings
X_dim = 1
z_dim = 1

#parameters that need to be tuned
g_dim = 64
batchsize = 200
lr = 1e-3
beta = 0.1
num_proximal = 200


#data
np.random.seed(0)
y = np.random.uniform(0,2,[50000])
z_pool = np.random.uniform(-1,1,[50000])

#functions to plot
def plot_hist(data,it,mb_size):
    bin=20  
    plt.plot([0.,0.,2.,2.],[0.,0.5,0.5,0.],color='darkred',lw=3,label = 'Ground Truth')
    plt.hist(data, bins=bin,normed=True,edgecolor='peru',color='papayawhip',label = 'Generated Data')
    if it=='Final Result':
        plt.legend(fontsize=15,loc = 2)
        plt.ylim([0,0.8])
    plt.title('Plot of  '+str(it), fontsize=25)
    plt.show()

#function to batch data. This is a pretty simple one, requiring N%batchsize==0
def next_batch(data,iter,batchsize):
  N=np.size(data,0)
  start=(iter*batchsize % N)
  return data[start:start+batchsize]

#functions for initialization
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    #return tf.random_normal(shape=size, stddev=xavier_stddev/3)+0.2
    return tf.random_normal(shape=size, stddev=xavier_stddev/3)
    
def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1)
  return tf.Variable(initial)




#build the model
z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, g_dim]),tf.float64)
G_b1 = bias_variable([g_dim])

G_W2 = tf.Variable(xavier_init([g_dim, X_dim]),tf.float64)
G_b2 = bias_variable([X_dim])
theta_G = [G_W1, G_b1, G_W2, G_b2]

def generator(z,G_W1, G_b1,G_W2, G_b2):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    return G_log_prob

X = generator(z,G_W1, G_b1,G_W2, G_b2)

P_vec = tf.placeholder(tf.float32, shape=[batchsize, z_dim])

G_loss = tf.reduce_sum(P_vec*X)

G_solver = (tf.train.RMSPropOptimizer(learning_rate=lr)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())




loss_list = []
one = np.ones([batchsize,])

# Traning
print('Start training...')
start_time = time.time()
for it in range(1000):
    #sample batches of data
    y_batch = next_batch(y,it,batchsize)
    z_batch = next_batch(z_pool,it,batchsize)
    
    x_batch = sess.run(X, feed_dict={z: np.expand_dims(z_batch,axis=1)})
    
    #compute the distance
    xtile = x_batch                                                                                                                
    ytile = np.expand_dims(y_batch,axis=0)                                                                                                                 
    deltaC = xtile - ytile                                                                                                                          
    C = deltaC*deltaC                                                                                                                            
    C = C/np.max(C)     
    
    #optimal map and WD
    #for larger problems, this part should be implemented by tensorflow
    P = ipot.ipot_WD(one,one,C,beta=beta,max_iter=num_proximal,return_loss=False)
    W = np.sum(P*C)
    loss_list.append(W)

    update = np.sum(P*deltaC, axis=1)
    
    #train
    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: np.expand_dims(z_batch,axis=1), P_vec: np.expand_dims(update,axis=1)}
    )    
    
    if it%100==0:
        print('Iter: ',it,'   WD: ',W,'   Time: ',time.time()-start_time)
        data_figure = sess.run(X, feed_dict={z: np.expand_dims(z_pool,axis=1)})
        plot_hist(data_figure,it,batchsize)


#plot the loss
plt.plot(loss_list)
plt.title('Plot of WD', fontsize=25)
plt.show()

#plot the generated result
plot_hist(data_figure,'Final Result',batchsize)
