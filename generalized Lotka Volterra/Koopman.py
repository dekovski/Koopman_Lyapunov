# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 11:18:45 2021

@author: shank
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy.polys.orderings import monomial_key
from sympy.polys.monomials import itermonomials
import sympy
from sympy import lambdify
from numpy.linalg import inv, eig, pinv, det
from scipy.linalg import svd, svdvals, sqrtm
from numpy import diag, dot, real, imag
import scipy.io
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib
import matplotlib.colors as mcolors
import winsound
from scipy.optimize import linprog

def dynamics(states,p):
    return np.multiply(p.rho.reshape(11,1), states) + np.multiply( np.dot(p.K, states), states)
                
"""
def sample_data(dic,N,T, fp, p, eps = 0.1):
    X = np.empty((T*N,11))
    Y = np.empty((T*N,11))
    P = 2*np.random.rand(N,11)-1
    P =  P/np.linalg.norm(P,axis=1).reshape(N,1)
    P = (fp + eps*np.random.rand(N,1)*P)
    idx = np.argwhere(P<0).T
    P[idx[0],idx[1]]=1e-5;
    Start = P

    for k in range(T):
        X[k*N:(k+1)*N,:] = P
        P = P + dt*dynamics(P.T,p).T
        Y[k*N:(k+1)*N,:] = P
    X = basis_fun(X.T)
    Y = basis_fun(Y.T)
    return X,Y, Start
    
    
def sample_trajectories(N,T, fp, p, in_plane = True, eps = 0.1):
    traj = np.empty((N,T,11))
    if(not(in_plane)):
        P = 2*np.random.rand(N,11)-1
        P =  P/np.linalg.norm(P,axis=1).reshape(N,1)
        P = (fp + eps*np.random.rand(N,1)*P)
        idx = np.argwhere(P<0).T
        P[idx[0],idx[1]]=1e-5;
        Start = P
    else:
        P = np.random.rand(N,11)*ssr_steady_states['E'] + np.random.rand(N,11)*ssr_steady_states['C']
        Start = p
        
    for k in range(T):
        traj[:,:,k] = P
        P = P + dt*dynamics(P.T,p).T
    return traj, Start
"""

def sample_data(N,T, fp_x, p, eps = 0.1):
    X = np.empty((N,T,11))
    P = 2*np.random.rand(N,11)-1; 
    #P = (fp_x + eps*P)
    P =  P/np.linalg.norm(P,axis=1).reshape(N,1); P = (fp_x + eps*P*np.random.rand(N,1))
    idx = np.argwhere(P<0).T
    P[idx[0],idx[1]]=1e-5;
    t = np.linspace(0,T/10,T)
    for i,ic in enumerate(P):
        X[i] = integrate.odeint(p.integrand, ic, t, atol=1e-12)
    return X, P

def sample_trajectories(N,T, fp_x, p, in_plane = True, eps = 0.1, eps2 = 0.9):
    X = np.empty((N,T,11))
    if(not(in_plane)):
        P = 2*np.random.rand(N,11)-1
        P =  P/np.linalg.norm(P,axis=1).reshape(N,1)
        P = (fp_x.T + eps*np.random.rand(N,1)*P)
        idx = np.argwhere(P<0).T
        P[idx[0],idx[1]]=1e-5;
        Start = P
    else:
        #P = eps*np.random.rand(N,1)*(fp1+fp2-fp_x) + (1-eps*np.random.rand(N,1))*fp_x
        P = eps2*np.random.rand(N,1)*(fp1 + fp2 - fp_x) + (1-eps*np.random.rand(N,1))*fp_x
        Start = P
        
    t = np.linspace(0,T/10,T)
    
    for i,ic in enumerate(P):
        X[i] = integrate.odeint(p.integrand, ic, t, atol=1e-12)
    return X, Start


def sample_2d_uniform(T, p):
    coord = np.linspace(0,1,100)
    a,b = np.meshgrid(coord, coord);
    a = a.reshape(-1,1);
    b = b.reshape(-1,1);    
    P = a*fp1 + b*fp2;
    Start = P
    
    t = np.linspace(0,T/10,T)
    X = np.empty((len(a),T,11))
    
    for i,ic in enumerate(list(P)):
        X[i] = integrate.odeint(p.integrand, ic, t, atol=1e-12)
    return X, Start


def basis_fun(X):
    ret = np.ones((len(monomials),X.shape[1]))
    ret[1:,:] = np.array(basis(*X)[1:])
    return ret

### Load saved data or do EDMD from scratch ###
"""
P = np.load('./saved/phi_stable.npy')
alpha = np.load('./saved/alpha.npy')
P_real = np.real(P)
P_imag = np.imag(P)
rho = np.load('./saved/rho.npy')
K = np.load('./saved/K.npy')
gamma = np.load('./saved/gamma.npy')
beta = np.load('./saved/beta.npy')
"""

def torch_f(X):
    f = torch.multiply(torch.tensor(rho.reshape(11,1)),X)+ torch.multiply(torch.matmul(torch.tensor(K),X),X)
    return f

def torch_V(X):
    ret = basis(*X)
    ret[0] = torch.ones(ret[1].shape, dtype=torch.float64)
    ret = torch.vstack(ret)
    E = torch.matmul(torch.tensor(P_real.T),ret)**2 + torch.matmul(torch.tensor(P_imag.T),ret)**2
    return torch.matmul(torch.tensor([alpha]),E)[0]

def torch_V_dot(X):
    dV = torch.autograd.functional.jacobian(torch_V,X)[0]
    f = torch_f(X)
    return torch.matmul(f.T,dV)[0]
    
def con_fun(X):
    X = torch.tensor(X.reshape((11,1)))
    return torch_V(X).data.numpy() - gamma
def con_jac(X):
    print("In con_jac : " + str(X.shape))
    X = torch.tensor(X.reshape((11,1)))
    ret = torch.autograd.functional.jacobian(torch_V,X)
    return ret[0].data.numpy()
def con_hess(X,V):
    X = torch.tensor(X.reshape((11,1)))
    ret = torch.autograd.functional.hessian(torch_V,X)
    return V[0]*ret[:,0,:,0].data.numpy()

def obj_fun(X):
    X = torch.tensor(X,dtype=torch.float64)
    ret = torch_V_dot(X)  - torch.tensor(beta)*(torch.tensor(gamma)-torch_V(X))
    return -ret.data.numpy()
def obj_jac(X):
    X = torch.tensor(X,dtype=torch.float64)
    ret = torch.autograd.functional.jacobian(torch_V_dot,X)  + torch.tensor(beta)*torch.autograd.functional.jacobian(torch_V,X)
    return -ret[0].data.numpy()
def obj_hess(X):
    X = torch.tensor(X.reshape((11,1)),dtype=torch.float64)
    ret = torch.autograd.functional.hessian(torch_V_dot,X)  + torch.tensor(beta)*torch.autograd.functional.hessian(torch_V,X)
    return -ret[:,0,:,0].data.numpy()

deg = 3
x = sympy.symbols('x1:12')
x_list = list(x)
monomials = sorted(itermonomials(x, deg), key=monomial_key('grlex', x))
basis = lambdify(x, monomials,"numpy")


### STABLE = 'A','C','D' (A and D are same!!!)
### UNSTABLE = 'B', 'E'
p = Params()
ssr_steady_states = get_all_stein_steady_states(p)
fp1 = ssr_steady_states['C'].T      #y-axis
fp2 = ssr_steady_states['E'].T      #x-axis
fp = fp2;
jac = np.diag(np.dot(p.K, fp) + p.rho) + np.dot(np.diag(fp),p.K)
e,v = eig(jac)
real(e)

###### Sample data close to the first fixed point #####
T = 100
N = 500
eps = 10
Traj,Start = sample_data(N,T,fp1, p, eps)
X = basis_fun(np.concatenate(Traj[:,0:-1,:],axis=0).T)
Y = basis_fun(np.concatenate(Traj[:,1:,:],axis=0).T)
#winsound.Beep(2500,500)

##### Sample data close to the second fixed point ####

T = 100
N = 500
eps = 15
Traj,Start = sample_data(N,T,fp2, p, eps)
X_ = basis_fun(np.concatenate(Traj[:,0:-1,:],axis=0).T)
Y_ = basis_fun(np.concatenate(Traj[:,1:,:],axis=0).T)
X = np.concatenate((X,X_),axis=1)
Y = np.concatenate((Y,Y_),axis=1)


### EDMD ###
A = np.matmul(Y,np.linalg.pinv(X)) # Can actually do this one data point at a time, useful for realitime update
mu_,phi_ = eig(A.T)
res_eig = np.abs(np.matmul(phi_.T,Y - np.matmul(A,X)))
res_eig = np.max(res_eig,axis=1)
#print(np.sort(res_eig))
top = 100
s = np.argsort(res_eig)#[0:top]
phi = phi_[:,s]
mu = mu_[s]

stable = np.argwhere(real(np.log(mu))<0).T[0]  
phi_stable = phi[:,stable] 
mu_stable = mu[stable]
res_eig_stable = np.abs(np.matmul(phi_stable.T,Y - np.matmul(A,X)))
res_eig_stable = np.max(res_eig_stable,axis=1)
s_stable = np.argsort(res_eig_stable)
res_eig_stable = np.sort(res_eig_stable)
phi_stable = phi_stable[:,s_stable]
mu_stable = mu_stable[s_stable]
plt.figure()
plt.plot((res_eig_stable))

"""
unstable = np.argwhere(real(np.log(mu))>0).T[0]  
phi_unstable = phi[:,unstable] 
mu_unstable = mu[unstable]
res_eig_unstable = np.abs(np.matmul(phi_unstable.T,Y - np.matmul(A,X)))
res_eig_unstable = np.max(res_eig_unstable,axis=1)
s_unstable = np.argsort(res_eig_unstable)
phi_unstable = phi_unstable[:,s_unstable]
mu_unstable = mu_unstable[s_unstable]
plt.figure()
plt.plot(np.sort(res_eig_unstable))
winsound.Beep(2500,500)
"""

"""
plt.plot(real(np.log(mu)))
## COMPARE KOOPMAN WITH USUAL FORWARD DYNAMICS ##

num_steps=T
Traj_s,S = sample_data(1,num_steps,fp,p,eps=5)

Xa = np.empty((X.shape[0],num_steps))
Xa[:,0] = basis_fun(S.T).T[0]
for i in range(num_steps-1):
    Xa[:,i+1]= np.matmul(A,Xa[:,i])
Traj_koop = Xa[11:0:-1,:]

plt.figure()
plt.title('Koopman prediction')
plt.plot(Traj_s[0],':',label='Actual')
plt.plot(num_steps*np.ones(11),fp,'*')
plt.plot(Traj_koop.T,label='Koopman')
#plt.legend()

####

#np.matmul(phi_stable.T,basis_fun(P.T))
Traj_s,S = sample_data(1,1000,fp,p,eps=15)
Xs = basis_fun(np.concatenate(Traj_s[:,0:-1,:],axis=0).T)
Ys = basis_fun(np.concatenate(Traj_s[:,1:,:],axis=0).T)
ret = np.matmul(phi_stable.T,Xs)
ret = real(ret*(ret.conj()))
ret = ret[1:].sum(axis=0)
plt.figure();
plt.plot(np.log10(ret))
#plt.figure(); plt.plot((ret[1:,0:200].T))
"""

def Lyap_2d(X,P):
    X_ = X[0].reshape((1,-1))
    Y_ = X[1].reshape((1,-1))
    sh = X[0].shape
    y = np.array([fp2]).T*X_ +  np.array([fp1]).T*Y_
    Y = basis_fun(y)
    ret = np.matmul(P.T,Y)
    ret = real(ret*(ret.conj()))
    ret = ret[0:].sum(axis=0) #np.matmul(alpha,ret[1:]) #ret[1:].sum(axis=0)
    return ret.reshape(sh)
    

### VISULAIZE ###
delta = 0.01
x_ = np.arange(-0.10, 1.1 + delta, delta)
y_ = np.arange(-0.10, 1.1 + delta, delta)

X_, Y_ = np.meshgrid(x_, y_)
#coeff = scipy.special.softmax(-1*res_eig_stable/res_eig_stable[0])
#P =  np.matmul(phi_stable,np.diag(coeff))
P = phi_stable
Z = np.log10(Lyap_2d(np.array([X_,Y_]),P))
levels =  MaxNLocator(nbins=100).tick_values(Z.min(), Z.max())
#np.array([-5. , -2.5,  -1.75, 0.2 ,  2.5,  5. ,  7.5, 10]) # MaxNLocator(nbins=5).tick_values(Z.min(), Z.max())

# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('hot')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
matplotlib.rcParams['pcolor.shading'] = 'auto'
fig, ax0 = plt.subplots()

im = ax0.pcolormesh(X_, Y_, Z, norm=norm) #cmap = cmap
fig.colorbar(im, ax=ax0)

con = plt.contour(X_,Y_, Z,  np.sort(np.hstack([levels])),linewidths=0.2, colors = 'k')
con.cmap

ax0.set_title('log(V(x))')

s = ssrParams(p, ssr_steady_states['A'], ssr_steady_states['C'])
ax1 = plot_ND_separatrix(p, s, ax=plt.gca(), save_plot=True, label='high-dimensional separatrix', color='red')

"""
scipy.io.savemat('phi_stable.mat', {'phi_stable': phi_stable})
scipy.io.savemat('mu_stable.mat', {'mu_stable': mu[stable]})
scipy.io.savemat('K.mat', {'K': p.K})
scipy.io.savemat('rho.mat', {'rho': p.rho})

#################################################

plt.figure();
V1 = [];
V2 = [];
num_steps = 200
num_traj = 50
alpha = coeff
#traj_s,S = sample_trajectories(num_traj, num_steps, fp1, p,  True, 0.7, 0.3)
traj_s,S = sample_2d_uniform(num_steps, p)
for i in range(len(traj_s)):
    traj_2d = p.project_to_2D(traj_s[i],fp2,fp1)
    plt.plot(traj_2d[0:200,0],traj_2d[0:200,1],'k',linewidth=0.2)
    if np.linalg.norm(traj_2d[-1,:] - np.array([1,0])) < np.linalg.norm(traj_2d[-1,:] - np.array([0,1])):
        plt.plot(traj_2d[0,0],traj_2d[0,1],'o',color='g', markersize=2)
    else:
        plt.plot(traj_2d[0,0],traj_2d[0,1],'o',color='m', markersize=2)
    ret = np.matmul(P.T,basis_fun(traj_s[i,:,:].T))
    ret = real(ret*(ret.conj()))
    ret = np.matmul(alpha,ret) #.sum(axis=0)
    V1.append(ret)  
    
plt.plot(0,1,'*',color='m');plt.plot(1,0,'*',color='g');plt.plot(0,0,'*',color='k')

Delta_V = np.array(V1); Delta_V = Delta_V[:,:,1:] - Delta_V[:,:,0:-1]
#A_ub = np.concatenate(Delta_V,axis=1).T
A_ub = Delta_V[:,:,0]
b = np.zeros(A_ub.shape[0])
c = np.ones(A_ub.shape[1])
#res = linprog(-c, A_ub=A_ub,b_ub=b.T,bounds = bounds,method='revised simplex'); print(res)

import polytope as pc

upper = np.ones(A_ub.shape[1])
lower = np.zeros(A_ub.shape[1])
bounds = [(u,l) for u,l in zip(lower,upper)]
box = pc.box2poly(bounds)

import time
tic = time.time()
intersection = box;
for k in range(Delta_V.shape[2]):
    print(k)
    A_ub = Delta_V[:,:,k]
    b = np.zeros(A_ub.shape[0])
    c = np.ones(A_ub.shape[1])
    p1 = pc.Polytope(np.vstack([A_ub, box.A]),np.hstack([b, box.b]))
    intersection_ = p1.intersect(intersection)
    if(pc.is_empty(intersection_)):
        print("terminating")
        break
    else:
        intersection = intersection_
toc = time.time()

####################################################
plt.figure()
traj_s,S = sample_trajectories(num_traj, num_steps, fp2, p, True, 0.7, 0.3)
for i in range(num_traj):
    traj_2d = p.project_to_2D(traj_s[i],fp2,fp1)
    plt.plot(traj_2d[:,0],traj_2d[:,1],'k',linewidth=0.5)
    if np.linalg.norm(traj_2d[-1,:] - np.array([1,0])) < np.linalg.norm(traj_2d[-1,:] - np.array([0,1])):
        plt.plot(traj_2d[0,0],traj_2d[0,1],'o',color='g')
    else:
        plt.plot(traj_2d[0,0],traj_2d[0,1],'o',color='m')
    plt.plot(traj_2d[-1,0],traj_2d[-1,1],'*',color='k')
    ret = np.matmul(phi_stable.T,basis_fun(traj_s[i,:,:].T))
    ret = real(ret*(ret.conj()))
    ret = ret[1:].sum(axis=0)
    V2.append(ret)
    
    
plt.figure();
plt.plot((np.array(V1).T)[:30])
"""

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

import polytope as pc
from scipy.optimize import linprog 
from matplotlib.animation import  PillowWriter

eps = 5

#X,Y,Start = sample_data(basis_fun,Num_samples,Traj_len,eps=eps)
# Initialize beta and increase it if the for-loop doesn't work as intended.
beta = 0.1
P = phi_stable
ms = mu_stable
# Initialize alpha set to unit-box as follows. 
# Will refine this alpha set by taking "intersection" in the for-loop.
alpha = np.ones((1,len(ms)+1)) 
upper = np.ones((len(ms)+1)); #upper[-1]=10;
lower = np.zeros((len(ms)+1))
bounds = [(u,l) for u,l in zip(lower,upper)]
box = pc.box2poly(bounds)
alpha_constraint = -np.ones(alpha.shape)
alpha_constraint[0,-1]=0
intersection = pc.Polytope(np.vstack([alpha_constraint,box.A]),np.hstack([np.array([-1]),box.b]))
Region_initial = pc.Region([intersection])

#sep = np.matmul(np.array([fp2,fp1]).T,np.array([ax1.lines[0].get_xdata(),ax1.lines[0].get_ydata()]))
#superlevel_constraint = pc.Polytope(np.vstack([-np.vstack([V_basis(basis_fun(sep),P),-np.ones((1,sep.shape[1]))]).T, box.A]), np.hstack([np.zeros((sep.shape[1])),box.b]))
superlevel_constraint = pc.Polytope(np.vstack([-np.vstack([V_basis(basis_fun(np.array([fp2]).T*0 +  np.array([fp1]).T*0),P),-np.ones((1,1))]).T, box.A]), np.hstack([np.zeros(1),box.b]))
      

import time

ax = None
tic = time.time()
alpha_history = []
data_points = 0
k=0
Samples = [] 
Bad_Samples = []
Traj_len = 20
Num_samples = 10
margin = 1 + 0
dt = 0.1*Traj_len/(Traj_len-1)

while (k<100):
    print("Iteration " + str(k))
    
    # Get the extreme values of each alpha_i in the current alpha-set.
    
    """
    #alpha = intersection.bounding_box[1]
    alpha = intersection.chebXc
    alpha = np.reshape(alpha,(1,-1))
    """
    
    Traj,Start = sample_data(Num_samples,Traj_len,fp1, p, eps)
    X = basis_fun(np.concatenate(Traj[:,0:-1,:],axis=0).T)
    Y = basis_fun(np.concatenate(Traj[:,1:,:],axis=0).T)
    Delta_V = (V_basis(Y,P) - V_basis(X,P))/dt
    Delta_V = np.vstack([Delta_V, np.zeros((1,Delta_V.shape[1]))])
    V_ = V_basis(X,P)
    V_ = np.vstack([V_, -margin*np.ones((1,V_.shape[1]))])
        
    A1 = (Delta_V + beta*V_).T
    A2 = V_.T
    b1 = np.zeros(A1.shape[0])
    b2 = np.zeros(A2.shape[0]) #+ margin
    A_ub = np.vstack([A1, A2, box.A])
    b_ub = np.hstack([b1, b2, box.b])
    p1 = pc.Polytope(A_ub,b_ub)
    
    #OR the point is outside the gamma-sublevel set
    """
    A = -V_.T
    b = np.zeros(A.shape[0])
    A_ub = np.vstack([A, box.A])
    b_ub = np.hstack([b, box.b])
    p2 = pc.Polytope(A_ub,b_ub)
    
    Region = pc.Region([p1,p2])
    """
    
    # Intersect previous alpha set, "intersection",  
    # with current alpha set "p1", which is a polytope.
    intersection_ = (p1.intersect(intersection.intersect(superlevel_constraint)))
    #intersection_ = Region & Region_initial
    
    
    # If this intersection is empty, try increasing beta, hope it works.
    if(pc.is_empty(intersection_)):
        print("Empty intersection for iteration " + str(k))
        Bad_Samples += [Start]
        continue
    
    intersection = intersection_
    #Region_initial = intersection_
    
    #print("Volume of alpha set " + str(k) + " = ")
    #print(p1.volume(intersection))
    Samples+=[Start]
    
    # Do some stuff to make cool animations
    alpha_history.append(intersection.chebXc)

    k = k+1
    
toc = time.time()


############################################
"""
superlevel_constraint = pc.Polytope(np.vstack([-np.vstack([V_basis(basis_fun(np.array([fp2]).T*0 +  np.array([fp1]).T*0),P),np.array([-1])]).T, box.A]), np.hstack([np.array([0]),box.b]))
temp = intersection
intersection = intersection.intersect(superlevel_constraint)
"""

T#raj,Start = sample_data(1,1,fp1, p, eps)
#X = basis_fun(np.concatenate(Traj,axis=0).T)
#V_ = V_basis(X,P)
#V_ = np.vstack([V_, -np.ones((1,V_.shape[1]))])
V_ = V_basis(basis_fun(np.array([fp2]).T*0 +  np.array([fp1]).T*0),P)
V_ = np.vstack([V_, -np.ones((1,V_.shape[1]))])
c = V_[:,0]
res = linprog(-c, A_ub=intersection.A,b_ub=intersection.b,bounds = bounds,method='revised simplex')
alpha = res.x
alpha1 = np.reshape(alpha,(1,-1))
alpha = intersection.chebXc
alpha2 = np.reshape(alpha,(1,-1))
alpha = alpha1 + 1e-5*(alpha2 - alpha1)
gamma = alpha[0,-1]
alpha = alpha[0,0:-1]
alpha[np.argmin(alpha)]=0
gamma_ref = np.sum(alpha*c[0:-1])

############################################
 
delta = 0.01
x_ = np.arange(-0.01, 1 + delta, delta)
y_ = np.arange(-0.01, 1 + delta, delta)

X_, Y_ = np.meshgrid(x_, y_)
coeff = alpha**(0.5)
P_ =  np.matmul(phi_stable[:,0:len(alpha)],np.diag(coeff))
#P = phi_stable[:,0:1200]
Z = np.log10(Lyap_2d(np.array([X_,Y_]),P_))
levels =  MaxNLocator(nbins=20).tick_values(Z.min(), Z.max())
#np.array([-5. , -2.5,  -1.75, 0.2 ,  2.5,  5. ,  7.5, 10]) # MaxNLocator(nbins=5).tick_values(Z.min(), Z.max())

# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('plasma')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
matplotlib.rcParams['pcolor.shading'] = 'auto'
fig, ax0 = plt.subplots()

im = ax0.pcolormesh(X_, Y_, Z, norm=norm, cmap = cmap)
fig.colorbar(im, ax=ax0)

con = plt.contour(X_,Y_, Z,  levels ,linewidths=0.2, colors = 'k')
con = plt.contour(X_,Y_, Z, [np.log10(gamma)], linewidths=1.2, colors = 'r')
con.collections[-1].set_label("$\gamma$-level set of V(x)")
con.cmap
con = plt.contourf(X_,Y_, Z, [Z.min(),np.log10(gamma)], hatches=[".."], alpha=0)
ax0.set_title('log(V(x))', fontsize=16)
ax0.set_xlabel("$\hat{x}_E$", fontsize=16)
ax0.set_ylabel("$\hat{x}_C$", fontsize=16)

s = ssrParams(p, ssr_steady_states['E'], ssr_steady_states['C'])
ax1 = plot_ND_separatrix(p, s, ax=plt.gca(), save_plot=True , label='high-dimensional separatrix' ,color='k')
matplotlib.rcParams["legend.loc"] = 'upper right'

plt.show()

############################################

V_trajs = []
V_basis_bad = []
START = []
bad_start = []
TRAJ = []
while(len(V_trajs) < 1000):
    Traj,Start = sample_data(1,1000,fp1, p, eps=10)
    V_ = V_basis(basis_fun(Traj[0].T),P)
    V_ = np.matmul(alpha,V_)
    if(V_[0]>gamma):
        continue
    else:
        V_trajs += [V_]
        START += [Traj[0,0,:]]
        TRAJ += [Traj]
        if(np.max(V_)>gamma):
            bad_start += [Traj[:,0]]
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
Jac = Candidate.jacobian(x_list)
Hess = sympy.hessian(Candidate, x_list)

states = sympy.Matrix(x)
f = np.multiply(p.rho.reshape(11,1), states) + np.multiply( np.dot(p.K, states), states)
Candidate_dot = sympy.Matrix(np.matmul(Jac,f)) + beta*Candidate

obj_fun_ = lambdify(x,Candidate_dot,"numpy")
obj_fun = lambda X : -obj_fun_(*X)[0]
obj_jac_ = lambdify(x,Candidate_dot.jacobian(x_list),"numpy")
obj_jac = lambda X : -obj_jac_(*X)[0]
obj_hess_ = lambdify((x,y),sympy.hessian(Candidate_dot, [x,y]),"numpy")
obj_hess = lambda X: -obj_hess_(X[0],X[1])

con_fun_ = lambdify((x,y), Candidate, "numpy")
con_fun = lambda X : con_fun_(X[0],X[1])[0]
con_jac_ = lambdify((x,y),Jac,"numpy")
con_jac = lambda X : con_jac_(X[0],X[1])[0]
con_hess_ = lambdify((x,y),Hess,"numpy")
con_hess = lambda X,V : V[0]*con_hess_(X[0],X[1])
