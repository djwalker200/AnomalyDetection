# coding=utf-8
"""
Anonymous author
part of codes are taken from gcpn/graphRNN's open-source code.
Description: load raw smiles, construct node/edge matrix.
"""

import sys
import os

import math
import numpy as np
import networkx as nx
import random
'''
from rdkit import Chem
from rdkit.Chem import rdmolops
'''
import torch
import torch.nn.functional as F

#Computes inverse of a lower triangular matrix
def invert(L):

    ''' 
    params:
    L - (batch x N x N) tensor of lower triangular matrices
    outputs:
    invL - (batch x N x N) tensor of matrix inverses
    '''
    n = L.size(-1)
    invL = torch.zeros_like(L)
    for j in range(0,n):
        invL[:,j,j] = 1.0/L[:,j,j]
        for i in range(j+1,n):
            S = 0.0
            for k in range(i+1):
                S = S - L[:,i,k]*invL[:,k,j].clone()

            invL[:,i,j] = S/L[:,i,i]

    return invL

#Computes determinant of a lower triangular matrix
def determinant(L):
    '''
    params:
    L - (batch x N x N) tensor of lower triangular matrices
    outputs:
    det - (batch x 1) tensor of determinant values
    '''
    diagonals = torch.diagonal(L,dim1=1,dim2=2)
    det = torch.prod(diagonals,dim=1)

    return det

def vector_to_Lower_Triangular(v):
    '''
    params:
    v - (batch x (D^2 + D) / 2) vector of values
    outputs:
    L - (batch x D x D) lower triangular matrix
    '''

    N = v.size(-1)
    batch_size = v.size(0)
    D = int((-1 + math.sqrt(1 + 8 * N)) / 2)

    L = torch.zeros((batch_size,D,D))

    counter = 0
    for i in range(D):
        for j in range(i + 1):
            L[:,i,j] = v[:,counter]
            counter += 1

    return L


def evaluate_probability_multi(x,L):
    '''
    Evaluates the probability of a multivariate Gaussian with mean 0
    params:
    x: input variables to the distribution of the form x = (x_deq - mean) (batch,D)
    L: Lower Triangular matrix predicted by MLP (batch,D,D)
    '''
    d = x.size(1)
    constant_pi = 3.1415926535

    if torch.cuda.is_available():
        L = L.cpu()
        x = x.cpu()
    
    invL  = invert(L)
    invLt = torch.transpose(invL,1,2)

    inv_cov_matrix = torch.einsum('bij, bjk-> bik', invLt, invL) #(batch,D,D)
    val = torch.einsum('bij,bj->bi',inv_cov_matrix,x)        #(batch,D)
    val = torch.einsum('bi,bi->b',x,val)        #(batch) 
    num = torch.exp(-0.5 * val)

    detL = determinant(L)
    det = detL * detL
    denom = torch.sqrt(math.pow(2 * constant_pi,d) * det) #(batch)

    return num / denom

def evaluate_probability(x,alpha):
    '''
    Evaluates the probability of a Gaussian with mean 0
    params:
    x: input variables to the distribution of the form x = (adj_deq - mean) (batch)
    alpha: standard deviation value predicted by MLP (batch)
    '''
    constant_pi = 3.1415926535

    alpha = torch.squeeze(alpha)
    value = torch.square(x / alpha)
    num = torch.exp(-0.5 * value)
    denom = alpha * math.sqrt(2 * constant_pi)

    return num / denom

def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)        
        
    print('set seed for random numpy and torch')


    










                      

                                