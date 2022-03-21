# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 11:18:45 2021

@author: shank
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sympy.polys.orderings import monomial_key
from sympy.polys.monomials import itermonomials
import sympy
from sympy import lambdify
from numpy.linalg import inv, eig, pinv, det
from scipy.linalg import svd, svdvals, sqrtm
from numpy import diag, dot, real, imag, log

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from matplotlib import rc
import matplotlib
rc('text',usetex=False)
import scipy

import torch
from model import *
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
from read_dataset import data_from_name
from tools import *

state_space_dim = 2
koopman_dim = 20
SEED = 1
DATASET = 'vanderpol'   #'lorenz'
STEPS = 1
STEPS_BACK = 1
BATCH = 640 #64
INIT_SCALE = 1
PRED_STEPS = 1 #15
EPS = -1
model = koopmanAE(state_space_dim, koopman_dim, STEPS, STEPS_BACK, 1)
print('koopmanAE')
device = torch.device('cpu')
model = model.to(device)
model.load_state_dict(torch.load('./model_last_b.pkl',map_location=torch.device('cpu')))
model.eval()

#******************************************************************************
# Load data
#******************************************************************************

TRAINDAT = [];
for traj in range(100):
    
    Xtrain, Xtest, Xtrain_clean, Xtest_clean = data_from_name('vanderpol',1000, EPS)
    
    #******************************************************************************
    # Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
    #******************************************************************************
    Xtrain = add_channels(Xtrain)
    Xtest = add_channels(Xtest)
    
    # transfer to tensor
    Xtrain = torch.from_numpy(Xtrain).float().contiguous()
    Xtest = torch.from_numpy(Xtest).float().contiguous()
    
    #******************************************************************************
    # Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
    #******************************************************************************
    Xtrain_clean = add_channels(Xtrain_clean)
    Xtest_clean = add_channels(Xtest_clean)
    
    # transfer to tensor
    Xtrain_clean = torch.from_numpy(Xtrain_clean).float().contiguous()
    Xtest_clean = torch.from_numpy(Xtest_clean).float().contiguous()
    
    #******************************************************************************
    # Create Dataloader objects
    #******************************************************************************
    trainDat = []
    start = 0
    for i in np.arange(STEPS,-1, -1):
        if i == 0:
            trainDat.append(Xtrain[start:].float())
        else:
            trainDat.append(Xtrain[start:-i].float())
        start += 1
    
    if traj == 0:
        TRAINDAT = trainDat;
    else:
        for i in range(len(trainDat)):
            TRAINDAT[i] = torch.cat((TRAINDAT[i],trainDat[i]));


train_data = torch.utils.data.TensorDataset(*TRAINDAT)
del(trainDat)
del(TRAINDAT)
train_loader = DataLoader(dataset = train_data,
                              batch_size = BATCH,
                              shuffle = True)

####################


def dynamics(x, y):
    x_dot =-y
    y_dot = x - y + (x**2)*y
    # unstable fp
    return x_dot, y_dot

def sample_data(dic,N,T, eps = 0.1):
    X = np.empty((T*N,2))
    Y = np.empty((T*N,2))
    #idx = np.random.permutation(limit_cycle.shape[1])[0:N]
    if(eps>=0):
        idx = np.random.choice(np.arange(limit_cycle.shape[1]),N)
        thetas = 2*np.pi*np.random.rand(N)
        P = limit_cycle[:,idx].T + eps*np.random.rand(N,2)*np.stack([np.sin(thetas),np.cos(thetas)],axis=1)
    else:
        P = 8*np.random.rand(N,2) - 4
    #P[0,:] = np.array([0,0])
    Start = P
    for k in range(T):
        X[k*N:(k+1)*N,:] = P
        P = P - dt*np.array(dynamics(P[:,0],P[:,1])).T
        Y[k*N:(k+1)*N,:] = P
    X = basis_fun(X.T)
    Y = basis_fun(Y.T)
    return X,Y, Start

def basis_fun(X):
    inp = torch.from_numpy(X.T).float().contiguous()
    Z = model.k_encoder(inp).data.numpy()
    return np.concatenate(Z).T

def V(X, v, alpha=[]):
    # v = phi_stable or phi_unstable
    X1 = X[0].reshape((1,-1))
    X2 = X[1].reshape((1,-1))
    sh = X[0].shape
    Z = basis_fun(np.array([X1[0],X2[0]]))
    ret = V_basis(Z,v)
    if(len(alpha)==0):
        alpha = np.ones((1,v.shape[1]))
    ret = np.matmul(alpha,ret)
    return ret.reshape(sh)

def V_basis(X,v):
    ret = np.matmul(v.T,X)
    ret = real(ret*(ret.conj()))
    return ret
    
limit_cycle = np.load('limit_cycle.npy')


##############################
""" EXTRACT EIGENFUNCTIONS """
##############################

model.eval()
X = model.k_encoder(train_loader.dataset[0:][0].to(device))[:,0,:]
Y = model.k_encoder(train_loader.dataset[0:][1].to(device))[:,0,:]
Koop_edmd = torch.linalg.lstsq(X,Y).solution.T.detach()
Koop_model = model.dynamics.dynamics.weight
Koop = Koop_model
A = Koop.data.cpu().numpy()
X = X.data.cpu().numpy().T
Y = Y.data.cpu().numpy().T
# Y = AX
mu_,phi_ = eig(A.T)

stable = np.argwhere(real(log(mu_))<0).T[0]  
phi_stable = phi_[:,stable] 
mu_stable = mu_[stable]
res_eig = np.abs(np.matmul(phi_stable.T,Y - np.dot(A,X)))
res_eig = np.max(res_eig,axis=1)
s = np.argsort(res_eig)
res_eig = np.sort(res_eig)
phi_stable = phi_stable[:,s]
mu_stable = mu_stable[s]

unstable = np.argwhere(real(log(mu_))>0).T[0]  
phi_unstable = phi_[:,unstable] 
mu_unstable = mu_[unstable]

print(res_eig)

#########################################
""" VISUALIZE INITIAL ESTIMATE OF ROA """
#########################################
matplotlib.rcParams['figure.dpi'] = 300   
#matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
delta = 0.01
x_ = np.arange(-4.0, 4.0, delta)
y_ = np.arange(-4.0, 4.0, delta)
X_, Y_ = np.meshgrid(x_, y_)
coeff = scipy.special.softmax(-1*res_eig/res_eig[0])
P =  np.matmul(phi_stable,np.diag(coeff)) #phi_stable
V_ = V(np.array([X_,Y_]),P)#,coeff**2))
#V_ = ((V_ - V_.min())/(V_.max() - V_.min()))**(0.2)
Z = np.log10(V_) 

levels =  MaxNLocator(nbins=20).tick_values(Z.min(), Z.max())
cmap = plt.get_cmap('plasma')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
matplotlib.rcParams['pcolor.shading'] = 'auto'
fig, ax0 = plt.subplots()
im = ax0.pcolormesh(X_, Y_, Z, norm=norm, cmap = cmap)
labels = ['-4', '-2' , '0', '2', '4']
fig.colorbar(im, ax=ax0)
con = plt.contour(X_,Y_, Z, [-5.5 , -4.5 , -4.  , -3.75, -3.5 , -3.375, -3.25,-3.125,
       -3, -2.75],linewidths=0.2, colors = 'k') #[0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
con.cmap
ax0.set_title('Levelsets of V(x)')
ax0.set_xticks(np.arange(-4,6,2))
ax0.set_xticklabels(labels)
ax0.set_yticks(np.arange(-4,6,2))
ax0.set_yticklabels(labels)
plt.savefig('test.png',dpi=300)

plt.figure()
for traj in range(1000):
    V_temp = V(data_from_name('vanderpol',int(1000/0.7), 0.5)[0].T,phi_stable,alpha)
    plt.plot((V_temp), color = mcolors.CSS4_COLORS['crimson'], linewidth=0.2, alpha = 0.4)
plt.show()


############################################
""" ESTIMATE FEASIBLE POLYTOPE FROM DATA """
############################################

import polytope as pc
from scipy.optimize import linprog 
from matplotlib.animation import  PillowWriter

dt = 0.01 
eps = 0.5

#X,Y,Start = sample_data(basis_fun,Num_samples,Traj_len,eps=eps)
# Initialize beta and increase it if the for-loop doesn't work as intended.
beta = 1
P = phi_stable

# Initialize alpha set to unit-box as follows. 
# Will refine this alpha set by taking "intersection" in the for-loop.
alpha = np.ones((1,len(mu_stable)+1)) 
upper = np.ones((len(mu_stable)+1))
lower = np.zeros((len(mu_stable)+1))
bounds = [(u,l) for u,l in zip(lower,upper)]
box = pc.box2poly(bounds)
alpha_constraint = -np.ones(alpha.shape)
alpha_constraint[0,-1]=0
intersection = pc.Polytope(np.vstack([alpha_constraint,box.A]),np.hstack([np.array([-1]),box.b]))
superlevel_constraint = pc.Polytope(np.vstack([-np.vstack([V_basis(basis_fun(np.array([-1,-1.8])),phi_stable),np.array([-1])]).T, box.A]), np.hstack([np.array([0]),box.b]))
      
import time

ax = None
fig,axis = plt.subplots()
writer =  PillowWriter(10)
writer.setup(fig,"Alpha_set.gif")
tic = time.time()
alpha_history = []
data_points = 0
k=0
Samples = [] 
Bad_Samples = []
Traj_len = 10
Num_samples = 50
margin = 1 + 1e-3
while (k<20):
    print("Iteration " + str(k))
    
    # Get the extreme values of each alpha_i in the current alpha-set.
    """
    #alpha = intersection.bounding_box[1]
    alpha = intersection.chebXc
    alpha = np.reshape(alpha,(1,-1))
    """
    X,Y,Start = sample_data(basis_fun,Num_samples,Traj_len,eps=eps)
    Delta_V = (V_basis(Y,P) - V_basis(X,P))/dt
    Delta_V = np.vstack([Delta_V, np.zeros((1,Delta_V.shape[1]))])
    V_ = V_basis(X,P)
    V_ = np.vstack([V_, -margin*np.ones((1,V_.shape[1]))])
    
    """
    delete_idx = []
    for i,c in enumerate(V_.T):
        res = linprog(c, A_ub=np.delete(V_.T,i,0),b_ub=np.zeros((V_.shape[1]-1,1)),bounds = bounds,method='revised simplex')
        if(-res.fun<0):
            delete_idx += [i]
    if(len(delete_idx)>0):
        Delta_V = np.delete(Delta_V.T,delete_idx,0).T
        V_ = np.delete(V_.T,delete_idx,0).T     
    """
    """
    delete_idx = []
    for i,c in enumerate(V_.T):
        temp = pc.Polytope(np.vstack([np.array([c]),box.A]),np.hstack([np.array([0]),box.b]))
        if(pc.is_empty(temp.intersect(superlevel_constraint))):
            #superlevel_constraint = pc.Polytope(np.vstack([superlevel_constraint.A, -temp.A]),np.hstack([superlevel_constraint.b,-temp.b]))
            delete_idx += [i]
    for i in reversed(delete_idx):
        Delta_V = np.delete(Delta_V.T,i,0).T
        V_ = np.delete(V_.T,i,0).T
    """
    
    A1 = (Delta_V + beta*V_).T
    A2 = V_.T
    b1 = np.zeros(A1.shape[0])
    b2 = np.zeros(A2.shape[0]) #+ margin
    A_ub = np.vstack([A1, A2, box.A])
    b_ub = np.hstack([b1, b2, box.b])
    p1 = pc.Polytope(A_ub,b_ub)
    
    # Intersect previous alpha set, "intersection",  
    # with current alpha set "p1", which is a polytope.
    intersection_ = p1.intersect(intersection.intersect(superlevel_constraint))
    
    # If this intersection is empty, try increasing beta, hope it works.
    if(pc.is_empty(intersection_)):
        print("Empty intersection for iteration " + str(k))
        Bad_Samples += [Start]
        #beta *= 1.01
        continue
    
    intersection = intersection_
    
    #print("Volume of alpha set " + str(k) + " = ")
    #print(pc.volume(intersection))
    Samples+=[Start]
    
    # Do some stuff to make cool animations
    alpha_history.append(intersection.chebXc)
    ax = intersection.project([1,4]).plot(ax, color = (k/999)*np.ones(3), alpha = 1.0, linestyle = '-', linewidth = 0.5, edgecolor = "red")
    axis.patches = ax.patches
    axis.text(0.75, 0.9, "Iteration: " + str(k), fontsize=12, color='r'); 
    #plt.show()
    writer.grab_frame()
    axis.clear()
    k = k+1
    
toc = time.time()
writer.finish()


############################################################
""" PICK A GOOD CANDIDATE ALPHA FROM THE FINAL ALPHA SET """
############################################################

#Can pick chebyshev center of the polytope as follows.
#alpha = intersection.chebXc 

#Or using linear progamming to optimally pick candidate alpha
# so that the corresponding gamma sublevel set is the smallest (or largest)
X,Start = sample_data(basis_fun,1,1,eps=eps) 
V_ = V_basis(X,phi_stable)
V_ = V_basis(basis_fun(np.array([-0.56,1.9])),phi_stable)
V_ = np.vstack([V_, -np.ones((1,V_.shape[1]))])
res = linprog(-V_[:,0], A_ub=intersection.A,b_ub=intersection.b,bounds = bounds,method='revised simplex')
alpha = res.x
alpha1 = np.reshape(alpha,(1,-1))
alpha = intersection.chebXc
alpha2 = np.reshape(alpha,(1,-1))
alpha = alpha1 + 0.001*(alpha2 - alpha1)

#######################################
""" VISUALIZE THE FINAL REFINED ROA """
#######################################

gamma = alpha[0,-1]
alpha = alpha[0,0:-1]
delta = 0.01
x_ = np.arange(-4.0, 4.0, delta)
y_ = np.arange(-4.0, 4.0, delta)
X_, Y_ = np.meshgrid(x_, y_)
Z = np.log10(V(np.array([X_,Y_]),phi_stable,alpha))
levels =  MaxNLocator(nbins=20).tick_values(Z.min(), Z.max())
cmap = plt.get_cmap('binary')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
matplotlib.rcParams['pcolor.shading'] = 'auto'
#matplotlib.rcParams['figure.dpi'] = 300   
fig, ax0 = plt.subplots()
im = ax0.pcolormesh(X_, Y_, Z, norm=norm, cmap = cmap)
fig.colorbar(im, ax=ax0)
#con = plt.contour(X_,Y_, Z, [np.log10(gamma)], linewidths=1, colors = 'k')
con.cmap
ax0.set_title('log(V(x)). Dashed line is gamma sublevel set')

for traj in TRAJ:
    plt.plot(traj[0,0:50],traj[1,0:50],linewidth=0.2, alpha=0.5, color = mcolors.CSS4_COLORS['darkmagenta']);

Start = np.vstack(START)
plt.plot(Start[:,0],Start[:,1], marker=".",color=mcolors.CSS4_COLORS['darkmagenta'], markersize=2, linestyle=None, linewidth=0)

labels = ['-4', '-2' , '0', '2', '4']
ax0.set_title('Forward-invariant set')
ax0.set_xticks(np.arange(-4,6,2))
ax0.set_xticklabels(labels)
ax0.set_yticks(np.arange(-4,6,2))
ax0.set_yticklabels(labels)

plt.show()
plt.savefig('Invariant_set.png',dpi=300)
Samples = np.vstack(Samples)
plt.scatter(Samples[:,0],Samples[:,1],marker='x',color='r')


#plt.plot(limit_cycle[0],limit_cycle[1],linewidth=0.5, color = 'k', marker='x')

V_trajs = []
V_basis_bad = []
START = []
bad_start = []
TRAJ = []
while(len(V_trajs) < 1000):
    traj = data_from_name('vanderpol',int(1000/0.7), 1)[0].T
    V_temp = V(traj,phi_stable,alpha)
    V_temp_bad = V_basis(basis_fun(traj),phi_stable)
    if(V_temp[0]>gamma):
        continue
    else:
        V_trajs += [V_temp]
        START += [traj[:,0]]
        TRAJ += [traj]
        if(np.max(V_temp)>gamma):
            V_basis_bad += [V_temp_bad]
            bad_start += [traj[:,0]]
        #plt.plot(V_temp, color = mcolors.CSS4_COLORS['crimson'], linewidth=0.5, alpha = 0.4)
        print(str(len(V_trajs)) + ' , ' + str(len(bad_start)))
    

plt.figure()
matplotlib.rcParams['figure.dpi'] = 300   
for i,traj in enumerate(V_trajs):
    plt.plot(traj, color = mcolors.CSS4_COLORS['darkmagenta'], linewidth=0.2, alpha = 0.4)
plt.title("1000 randomly sampled trajectories \n starting inside the $\gamma$-sublevel set")
plt.plot(gamma*np.ones(traj.shape),'k--')
plt.show()

######################################
""" FIND COUNTER EXAMPLE OPTIMALLY """
######################################

mo = sympy.Matrix(monomials)
P_real = real(P)
P_imag = imag(P)
E = np.matmul(P_real.T,mo)**2 + np.matmul(P_imag.T,mo)**2
Candidate = sympy.Matrix(np.matmul(alpha,E) - gamma) 
Jac = Candidate.jacobian([x,y])
Hess = sympy.hessian(Candidate, [x,y])

f = sympy.Matrix([y,    -x + y - (x**2)*y])
Candidate_dot = sympy.Matrix(np.matmul(Jac,f)) + beta*Candidate

obj_fun_ = lambdify((x,y),Candidate_dot,"numpy")
obj_fun = lambda X : -obj_fun_(X[0],X[1])[0]
obj_jac_ = lambdify((x,y),Candidate_dot.jacobian([x,y]),"numpy")
obj_jac = lambda X : -obj_jac_(X[0],X[1])[0]
obj_hess_ = lambdify((x,y),sympy.hessian(Candidate_dot, [x,y]),"numpy")
obj_hess = lambda X: -obj_hess_(X[0],X[1])

con_fun_ = lambdify((x,y), Candidate, "numpy")
con_fun = lambda X : con_fun_(X[0],X[1])[0]
con_jac_ = lambdify((x,y),Jac,"numpy")
con_jac = lambda X : con_jac_(X[0],X[1])[0]
con_hess_ = lambdify((x,y),Hess,"numpy")
con_hess = lambda X,V : V[0]*con_hess_(X[0],X[1])


from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.optimize import Bounds

bounds = Bounds([-3, -3], [3, 3])
nonlinear_constraint = NonlinearConstraint(con_fun, -np.inf, 0, jac=con_jac, hess=con_hess)
x0 = np.array([1, 0])
res = minimize(obj_fun, x0, method='trust-constr', jac=obj_jac, hess=obj_hess,
               constraints=[nonlinear_constraint],
               options={'xtol': 1e-10, 'gtol': 1e-10, 'verbose': 1}, bounds=bounds)

# Negative res.fun means successfully found a Lyapunov function with no bad violation