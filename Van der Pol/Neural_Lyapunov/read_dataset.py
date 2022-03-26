import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

import torch

#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name,Traj_len=100, eps=2):
    if name == 'lorenz':
        return lorenz()  
    elif name == 'vanderpol':
        return vanderpol(Traj_len, eps)
    else:
        raise ValueError('dataset {} not recognized'.format(name))



def rescale(Xsmall, Xsmall_test):
    #******************************************************************************
    # Rescale data
    #******************************************************************************
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()
    
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test


def lorenz():

    def sol(x, y, z, s=10, r=28, b=2.667):
        '''
        Given:
           x, y, z: a point of interest in three dimensional space
           s, r, b: parameters defining the lorenz attractor
        Returns:
           x_dot, y_dot, z_dot: values of the lorenz attractor's partial
               derivatives at the point x, y, z
        '''
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot    
        
    dt = 0.01
    num_steps = 1000
    
    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    
    # Set initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = sol(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
        
    X = np.stack([xs,ys,zs],axis=1)
    Xclean = X.copy()
    
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    # split into train and test set 
    N = X.shape[0]
    X_train = X[0:int(0.7*N)]
    X_test = X[int(0.7*N):]
    X_train_clean = Xclean[0:int(0.7*N)]
    X_test_clean = Xclean[int(0.7*N):]  
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean



def vanderpol(Traj_len, eps):

    def sol(x, y):
        '''
        Given:
           x, y: a point of interest in three dimensional space
        Returns:
           x_dot, y_dot: values of the vanderpol attractor's partial
               derivatives at the point x, y
        '''
        x_dot =y
        y_dot = -x + y - (x**2)*y
        # unstable fp
        return x_dot, y_dot
        
    dt = 0.01
    num_steps = Traj_len
    
    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    
    # Set initial values
    if(eps>=0):
      limit_cycle = np.load('./data/limit_cycle.npy')
      idx = np.random.choice(np.arange(limit_cycle.shape[1]))
      theta = 2*np.pi*np.random.rand()
      xs[0], ys[0] = limit_cycle[:,idx] + np.random.rand()*eps*np.array([np.sin(theta),np.cos(theta)]) 
    else:
      xs[0], ys[0] = 8*np.random.rand(2)-4
    
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot = sol(xs[i], ys[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        
    X = np.stack([xs,ys],axis=1)
    Xclean = X.copy()
    
    
    # scale 
    # commented by shankar June 22
    """
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1
    """
    
    # split into train and test set 
    N = X.shape[0]
    X_train = X[0:int(0.7*N)]
    X_test = X[int(0.7*N):]
    X_train_clean = Xclean[0:int(0.7*N)]
    X_test_clean = Xclean[int(0.7*N):]  
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean