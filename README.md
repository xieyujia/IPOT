# IPOT
Implementation of Inexact Proximal point method for Optimal Transport from paper "A Fast Proximal Point Method for Computing Exact Wasserstein Distance" (https://arxiv.org/abs/1802.04307).

--------------------
Package dependencies
--------------------

It requires the following Python packages:

Python 3.5
numpy
matplotlib 
pot 0.4.0
tensorflow 1.1.0

-------------------
Included modules
--------------------

ipot.py: include function ipot_WD that computes the Wasserstein distance, a function ipot_barycenter that computes the Wasserstein barycenter.

ipot_demo.py: Demo for computing Wasserstein distance using ipot.ipot_WD.

learning_demo.py: 1D demo for learning generative model using ipot.ipot_WD. This file is the only one that needs Tensorflow.

barycenter_demo.py: Demo for computing Wasserstein barycenter.

-------------------
How to run the code
-------------------
1. To compute Wasserstein distance:

python ipot_demo.py

2. To learn 1D generative model:

python learning_demo.py

3. To compute Wasserstein barycenter:

python barycenter_demo.py

