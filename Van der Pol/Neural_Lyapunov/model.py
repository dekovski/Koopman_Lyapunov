# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:02:50 2022

@author: shank
"""


#==============================================================================
# Define Model
#==============================================================================

from torch import nn
import torch


def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega
    
    
class koopmanEncoder(nn.Module):
    def __init__(self, state_space_dim, koopman_dim):
        super(koopmanEncoder, self).__init__()
        self.N = state_space_dim
        self.h = koopman_dim #int(2*self.N)
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, self.h)
        self.fc2 = nn.Linear(self.h, self.h)
        self.fc3 = nn.Linear(self.h, koopman_dim)
        self.fc4 = nn.Linear(koopman_dim, koopman_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.N)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))        
        x = self.tanh(self.fc3(x)) #added tanh
        x = self.fc4(x)
        return x


class koopmanDecoder(nn.Module):
    def __init__(self, state_space_dim, koopman_dim):
        super(koopmanDecoder, self).__init__()
        
        self.b = koopman_dim
        self.n = state_space_dim
        self.h = koopman_dim #int(self.b/2)
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.b,self.h)
        self.fc2 = nn.Linear(self.h,self.h)
        self.fc3 = nn.Linear(self.h, state_space_dim)
        self.fc4 = nn.Linear(state_space_dim, state_space_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)          

    def forward(self, x):
        x = x.view(-1, 1, self.b)
        x = self.tanh(self.fc1(x)) 
        x = self.tanh(self.fc2(x)) 
        x = self.tanh(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1,1,self.n,1)
        return x


class dynamics(nn.Module):
    def __init__(self, koopman_dim, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(koopman_dim, koopman_dim, bias=False)
        self.dynamics.weight.data = gaussian_init_(koopman_dim, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

        
    def forward(self, x):
        x = self.dynamics(x)
        return x


class dynamics_back(nn.Module):
    def __init__(self, koopman_dim, omega):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(koopman_dim, koopman_dim, bias=False)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())     

    def forward(self, x):
        x = self.dynamics(x)
        return x

    
class koopmanAE(nn.Module):
    def __init__(self, state_space_dim, koopman_dim, steps, steps_back, init_scale=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.k_encoder = koopmanEncoder(state_space_dim,koopman_dim)
        self.dynamics = dynamics(koopman_dim, init_scale)
        self.backdynamics = dynamics_back(koopman_dim, self.dynamics)
        self.k_decoder = koopmanDecoder(state_space_dim,koopman_dim)

    def forward(self, x, mode='forward'):
        out = []
        out_back = []
        z = self.k_encoder(x.contiguous())
        q = z.contiguous()
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(q) #out.append(self.k_decoder(q))
            out.append(z.contiguous()) 
            return out, out_back    

        if mode == 'backward':
            for _ in range(self.steps_back):
                q = self.backdynamics(q)
                out_back.append(q) #out_back.append(self.k_decoder(q))
            out_back.append(z.contiguous())
            return out, out_back
