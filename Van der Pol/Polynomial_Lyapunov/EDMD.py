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
from numpy import diag, dot, real, imag
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


def dynamics(x, y):
    x_dot =-y
    y_dot = x - y + (x**2)*y
    # unstable fp
    return x_dot, y_dot

def sample_data(dic,N,T, eps = 0.1):
    X = np.empty((T*N,2))
    Y = np.empty((T*N,2))
    #idx = np.random.permutation(limit_cycle.shape[1])[0:N]
    idx = np.random.choice(np.arange(limit_cycle.shape[1]),N)
    thetas = 2*np.pi*np.random.rand(N)
    P = limit_cycle[:,idx].T + eps*np.random.rand(N,2)*np.stack([np.sin(thetas),np.cos(thetas)],axis=1)
    Start = P
    for k in range(T):
        X[k*N:(k+1)*N,:] = P
        P = P - dt*np.array(dynamics(P[:,0],P[:,1])).T
        Y[k*N:(k+1)*N,:] = P
    X = basis_fun(X.T)
    Y = basis_fun(Y.T)
    return X,Y, Start

def basis_fun(X):
    ret = np.ones((len(monomials),X.shape[1]))
    ret[1:,:] = np.array(basis(X[0],X[1])[1:])
    return ret

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
    

deg = 6
limit_cycle = np.load('./saved/limit_cycle.npy')

x, y = sympy.symbols('x y', real=True)
monomials = sorted(itermonomials([x, y], deg), key=monomial_key('grlex', [y, x]))
basis = lambdify((x,y),monomials,"numpy")

############################
""" SAMPLE TRAJECTORIES """
############################

dt = 0.01 
T = 500
N = 1000
eps = 0.5

X,Y,Start = sample_data(basis_fun,N,T,eps=eps)
X_test,Y_test, _ = sample_data(basis_fun,N,T,eps=eps)

plt.figure()
plt.plot(X[1],X[2],'r.',markersize=0.1)
plt.plot(Start[:,0],Start[:,1], 'g.',markersize=2)


####################
""" PERFORM EDMD """
####################

Fphi = np.eye(X.shape[0])
Xtr = X.copy()
Ytr = Y.copy()


while(Xtr.shape[0]>10):
    A = np.matmul(Ytr,np.linalg.pinv(Xtr)) # Can actually do this one data point at a time, useful for realitime update
    mu,phi = eig(A.T)
    res_eig = np.abs(np.matmul(phi.T,Ytr - np.matmul(A,Xtr)))
    res_eig = np.max(res_eig,axis=1)
    print(np.sort(res_eig))
    s = np.argsort(res_eig)
    phi = phi[:,s]
    m = Xtr.shape[0]
    m = int(m/2)
    phi = phi[:,0:m]
    Fphi = np.matmul(Fphi,phi)
    Xtr = np.matmul(Fphi.T,X)
    Ytr = np.matmul(Fphi.T,Y)
    
A = np.matmul(Ytr,np.linalg.pinv(Xtr)) # Can actually do this one data point at a time, useful for realitime update
mu,phi = eig(A.T)
K = np.diag(mu)
phi = np.matmul(Fphi,phi)
res_eig = np.abs(np.matmul(phi.T,Y) - np.matmul(K.T,np.matmul(phi.T,X)))
res_eig = np.max(res_eig,axis=1)
print(np.sort(res_eig))

s = np.argsort(real(np.log(mu)))
mu = mu[s]
phi = phi[:,s]

stable = np.argwhere(real(np.log(mu))<0).T[0]  
phi_stable = phi[:,stable] 
mu_stable = mu[stable]

unstable = np.argwhere(real(np.log(mu))>0).T[0]  
phi_unstable = phi[:,unstable] 
mu_stable = mu[stable]
print(np.log(mu))


#########################################
""" VISUALIZE INITIAL ESTIMATE OF ROA """
#########################################

delta = 0.01
x_ = np.arange(-4.0, 4.0, delta)
y_ = np.arange(-4.0, 4.0, delta)
X_, Y_ = np.meshgrid(x_, y_)
P = phi_stable[:,0:-1]
Z =  np.log10(V(np.array([X_,Y_]),P))
levels =  MaxNLocator(nbins=10).tick_values(Z.min(), Z.max())
cmap = plt.get_cmap('hot')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
matplotlib.rcParams['pcolor.shading'] = 'auto'
fig, ax0 = plt.subplots()
im = ax0.pcolormesh(X_, Y_, Z, norm=norm) #cmap = cmap
fig.colorbar(im, ax=ax0)
ax0.set_title('log(V(x))')
#plt.plot(limit_cycle[0],limit_cycle[1],linewidth=0.5)

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
superlevel_constraint = pc.Polytope(np.vstack([-np.vstack([V_basis(basis_fun(np.array([[-1],[-1.8]])),phi_stable),np.array([-1])]).T, box.A]), np.hstack([np.array([0]),box.b]))
      
import time

ax = None
fig,axis = plt.subplots()
writer =  PillowWriter(10)
writer.setup(fig,"Alpha_set1.gif")
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
V_ = V_basis(basis_fun(np.array([[-0.56],[1.9]])),phi_stable)
V_ = np.vstack([V_, -np.ones((1,V_.shape[1]))])
res = linprog(V_[:,0], A_ub=intersection.A,b_ub=intersection.b,bounds = bounds,method='revised simplex')
alpha = res.x
alpha1 = np.reshape(alpha,(1,-1))
alpha = intersection.chebXc
alpha2 = np.reshape(alpha,(1,-1))
alpha = alpha1 + 0.001*(alpha2 - alpha1)

gamma = alpha[0,-1]
alpha = alpha[0,0:-1]

## SAVE IF NEEDED ###
""" 
scipy.io.savemat('./saved/phi_stable.mat', {'phi_stable': phi_stable})
scipy.io.savemat('./saved/mu_stable.mat', {'mu_stable': mu_stable})
scipy.io.savemat('./saved/alpha.mat', {'alpha': alpha})
scipy.io.savemat('./saved/gamma.mat', {'gamma': gamma})
scipy.io.savemat('./saved/beta.mat', {'beta': beta})
"""

#######################################
""" VISUALIZE THE FINAL REFINED ROA """
#######################################

delta = 0.01
x_ = np.arange(-4.0, 4.0, delta)
y_ = np.arange(-4.0, 4.0, delta)
X_, Y_ = np.meshgrid(x_, y_)
Z = np.log10(V(np.array([X_,Y_]),P,alpha))
levels =  MaxNLocator(nbins=100).tick_values(Z.min(), Z.max())
cmap = plt.get_cmap('hot')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
matplotlib.rcParams['pcolor.shading'] = 'auto'
fig, ax0 = plt.subplots()
im = ax0.pcolormesh(X_, Y_, Z, norm=norm) #cmap = cmap
fig.colorbar(im, ax=ax0)
con = plt.contour(X_,Y_, Z, [np.log10(gamma)], linewidths=1, colors = 'k')
con.cmap
ax0.set_title('log(V(x)). Dashed line is gamma sublevel set')
#plt.plot(limit_cycle[0],limit_cycle[1],linewidth=0.5)

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

x0 = np.array([1, 0])
bounds = Bounds([-3, -3], [3, 3])
nonlinear_constraint = NonlinearConstraint(con_fun, -np.inf, 0, jac=con_jac, hess=con_hess)
res = minimize(obj_fun, x0, method='trust-constr', jac=obj_jac, hess=obj_hess,
               constraints=[nonlinear_constraint],
               options={'xtol': 1e-10, 'gtol': 1e-10, 'verbose': 1}, bounds=bounds)

print("Negative solution res.fun means no bad examples found")
