# coding=utf-8
"""
Anonymous author
"""
import os
import sys
import numpy as np
import math
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from FlowCore import NetModel



class GraphFlowModel(nn.Module):
    """
    Reminder:
        self.args: deq_coeff
                   deq_type

    Args:

    
    Returns:

    """
    def __init__(self, max_size, node_dim, args):
        super(GraphFlowModel, self).__init__()
        self.max_size = max_size
        self.node_dim = node_dim
        self.args = args

        ###Flow hyper-paramters
        self.num_flow_layer = self.args.num_flow_layer
        self.nhid = self.args.nhid
        self.nout = self.args.nout
      

        self.dp = False
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 1:
            self.dp = True
            print('using %d GPUs' % num_gpus)
        
        self.flow_core = NetModel(num_flow_layer = self.num_flow_layer,
                                       graph_size=self.max_size,
                                       node_dim=self.node_dim,
                                       args=self.args,
                                       nhid=self.nhid,
                                       nout=self.nout)
        if self.dp:
            self.flow_core = nn.DataParallel(self.flow_core)
         
        
        


    def forward(self, inp_node_features, inp_adj_features):
        """
        Args:
            inp_node_features: (BATCH_SIZE, N, NODE_FEATURES)
            inp_adj_features: (BATCH_SIZE, N, N)

        Returns:
            z: [(BATCH_SIZE, node_num* NODE_FEATURES), (BATCH_SIZE, edge_num)]
            logdet:  ([B], [B])        
        """
        #TODO: add dropout/normalize


        #inp_node_features_cont = inp_node_features #(B, N, NODE_FEATURES) #! this is buggy. shallow copy
        inp_node_features_deq = inp_node_features.clone() #(BATCH, N, NODE_FEATURES)

        
        inp_adj_features_deq = inp_adj_features.clone() #(BATCH,N,N)
        inp_adj_features_deq = inp_adj_features_deq.contiguous() 


        '''
        if self.args.deq_type == 'random':
            #TODO: put the randomness on GPU.!
            
            #CUDA MODIFICATION
            
            if self.args.cuda:
                inp_node_features_deq += self.args.deq_coeff * torch.rand(inp_node_features_deq.size()).cuda() #(BATCH_SIZE, N, NODE_FEATURES)
                inp_adj_features_deq += self.args.deq_coeff * torch.rand(inp_adj_features_deq.size()).cuda() #(BATCH_SIZE, N, N)
            else:
                
                inp_node_features_deq += self.args.deq_coeff * torch.rand(inp_node_features_deq.size()) #(BATCH_SIZE, N, NODE_FEATURES)
                inp_adj_features_deq += self.args.deq_coeff * torch.rand(inp_adj_features_deq.size()) #(BATCH_SIZE, N, N)
            
        elif self.args.deq_type == 'variational':
            #TODO: add variational deq.
            raise ValueError('current unsupported method: %s' % self.args.deq_type)
        else:
            raise ValueError('unsupported dequantization type (%s)' % self.args.deq_type)

        '''
        
        node_losses,edge_losses = self.flow_core(inp_node_features, inp_adj_features, 
                                   inp_node_features_deq, inp_adj_features_deq)
                                   
        if self.args.deq_type == 'random':
            return node_losses, edge_losses

        elif self.args.deq_type == 'variational':
            #TODO: try variational dequantization
            return node_losses,edge_losses, deq_logp


          
        

        
        

