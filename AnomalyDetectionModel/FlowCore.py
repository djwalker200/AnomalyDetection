#coding: utf-8
# Anonymous author


import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from time import time

def Linear(in_features, out_features, bias=True):

    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class RelationGraphConvolution(nn.Module):
    """
    Relation GCN layer. 
    """

    def __init__(self, in_features, out_features, dropout=0.0, use_relu=True, bias=False):
        '''
        :param in/out_features: scalar of channels for node embedding
        :param edge_dim: dim of edge type, virtual type not included
        '''
        super(RelationGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        if use_relu:
            self.act = nn.ReLU()
        else:
            self.act = None

        self.weight = nn.Parameter(torch.FloatTensor(
            self.in_features, self.out_features)) 
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(
                self.out_features)) 
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, adj):
        '''
        :param x: (batch, N, NODE_FEATURES)
        :param adj: (batch, N, N)
        
        :return:
        updated x with shape (batch, N, d)
        '''
        

        x = F.dropout(x, p=self.dropout, training=self.training)  # (batch, N, d)

        batch_size = x.size(0)

        # transform




        support = torch.einsum('bid, dh-> bih', x, self.weight) # (batch,N,d) x (d,out_feat) -> (batch,N,n_out)

        output = torch.einsum('bij, bjh-> bih', adj, support)  # (batch,N,N) x (batch,N,n_out) = (batch,N,n_out)

    

        if self.bias is not None:
            output += self.bias
        if self.act is not None:
            output = self.act(output)  
        output = output.view(batch_size, x.size( 
            1), self.out_features)  # (b, N, n_out)

        node_embedding = output

        return node_embedding

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RGCN(nn.Module):
    def __init__(self, nfeat, nhid=128, nout=128, num_layers=3, dropout=0.0, normalization=False,gcn_bias=False):
        '''
        :num_layers: the number of layers in each R-GCN
        '''
        super(RGCN, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.nout = nout
        self.num_layers = num_layers

        self.dropout = dropout
        self.normalization = normalization

        self.emb = Linear(nfeat, nfeat, bias=True) 
        #self.bn_emb = nn.BatchNorm2d(8)

        self.layer_weights = nn.Parameter(torch.FloatTensor(1,self.num_layers)) 

        self.gc1 = RelationGraphConvolution(nfeat, nhid, use_relu=True, dropout=self.dropout, bias = gcn_bias)
        # if self.normalization:
        #    self.bn1 = nn.BatchNorm2d(nhid)

        self.gc2 = nn.ModuleList([RelationGraphConvolution(nhid, nhid,use_relu=True, dropout=self.dropout, bias= gcn_bias)
                                  for i in range(self.num_layers-2)])
        # if self.normalization:
        #    self.bn2 = nn.ModuleList([nn.BatchNorm2d(nhid) for i in range(self.num_layers-2)])

        self.gc3 = RelationGraphConvolution(nhid, nout, use_relu=False, dropout=self.dropout, bias= gcn_bias)
        # if self.normalization
        #    self.bn3 = nn.BatchNorm2d(nout)

    def forward(self, x, adj):
        '''
        :param x: (batch, N, d)
        :param adj: (batch, N, N)
        :return:
        '''

        # TODO: Add normalization for adacency matrix
        # embedding layer
        #x = self.emb(x)


        # if self.normalization:
        #    x = self.bn_emb(x.transpose(0, 3, 1, 2))
        #    x = x.transpose(0, 2, 3, 1)



        # first GCN layer
        x1 = self.gc1(x, adj)    #(batch,N,nhid)



        # if self.normalization:
        #    x = self.bn1(x.transpose(0, 3, 1, 2))
        #    x = x.transpose(0, 2, 3, 1)

        # hidden GCN layer(s)
        x2 = self.gc2[0](x1, adj)  # (batch, N, nhid)

            # if self.normalization:
            #    x = self.bn2[i](x.transpose(0, 3, 1, 2))
            #    x = x.transpose(0, 2, 3, 1)


        # last GCN layer
        x3 = self.gc3(x2, adj)  # (batch, N, nout)


        x = torch.stack((x1,x2,x3))

        x = torch.einsum('if, fbnk-> ibnk',self.layer_weights,x)
        x = torch.squeeze(x,dim=0)
        # check here: bn for last layer seem to be necessary
        #x = self.bn3(x.transpose(0, 3, 1, 2))
        #x = x.transpose(0, 2, 3, 1)

        # return node embedding
        return x

# TODO: Try sample dependent initialization.!!
# TODO: Try different st function (sigmoid, softplus, exp, spine, flow++)
class ST_Net_Sigmoid(nn.Module):
    def __init__(self, args, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True, scale_weight_norm=False, sigmoid_shift=2.0, apply_batch_norm=False):
        super(ST_Net_Sigmoid, self).__init__()
        self.num_layers = num_layers  # unused
        self.args = args
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.apply_batch_norm = apply_batch_norm
        self.scale_weight_norm = scale_weight_norm
        self.sigmoid_shift = sigmoid_shift

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim, bias=bias)

        if self.apply_batch_norm:
            self.bn_before = nn.BatchNorm1d(input_dim)
        if self.scale_weight_norm:
            self.rescale1 = nn.utils.weight_norm(Rescale())
            self.rescale2 = nn.utils.weight_norm(Rescale())

        else:
            self.rescale1 = Rescale()
            self.rescale2 = Rescale()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def forward(self, x):
        '''
        :param x: (batch, input_dim)
        :return: weight and bias for affine operation
        '''


        if self.apply_batch_norm:
            x = self.bn_before(x)

        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        #Returns NN output of shape (batch,output_dim)
        return x

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear2.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.)
            nn.init.constant_(self.linear2.bias, 0.)




class ST_Net_Exp(nn.Module):
    def __init__(self, args, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True, scale_weight_norm=False, sigmoid_shift=2., apply_batch_norm=False):
        super(ST_Net_Exp, self).__init__()
        self.num_layers = num_layers  # unused
        self.args = args
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.apply_batch_norm = apply_batch_norm
        self.scale_weight_norm = scale_weight_norm
        self.sigmoid_shift = sigmoid_shift

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim*2, bias=bias)

        if self.apply_batch_norm:
            self.bn_before = nn.BatchNorm1d(input_dim)
        if self.scale_weight_norm:
            self.rescale1 = nn.utils.weight_norm(Rescale())
            #self.rescale2 = nn.utils.weight_norm(Rescale())

        else:
            self.rescale1 = Rescale()
            #self.rescale2 = Rescale()

        self.tanh = nn.Tanh()
        #self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear2.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.)
            nn.init.constant_(self.linear2.bias, 0.)

    def forward(self, x):
        '''
        :param x: (batch * repeat_num for node/edge, emb)
        :return: w and b for affine operation
        '''
        if self.apply_batch_norm:
            x = self.bn_before(x)

        x = self.linear2(self.tanh(self.linear1(x)))
        #x = self.rescale1(x)
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.rescale1(torch.tanh(s))
        #s = self.sigmoid(s + self.sigmoid_shift)
        #s = self.rescale2(s) # linear scale seems important, similar to learnable prior..
        return s, t


class ST_Net_Softplus(nn.Module):
    def __init__(self, args, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True, scale_weight_norm=False, sigmoid_shift=2., apply_batch_norm=False):
        super(ST_Net_Softplus, self).__init__()
        self.num_layers = num_layers  # unused
        self.args = args
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.apply_batch_norm = apply_batch_norm
        self.scale_weight_norm = scale_weight_norm
        self.sigmoid_shift = sigmoid_shift

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.linear3 = nn.Linear(hid_dim, output_dim*2, bias=bias)

        if self.apply_batch_norm:
            self.bn_before = nn.BatchNorm1d(input_dim)
        if self.scale_weight_norm:
            self.rescale1 = nn.utils.weight_norm(Rescale_channel(self.output_dim))

        else:
            self.rescale1 = Rescale_channel(self.output_dim)

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        #self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear3.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.)
            nn.init.constant_(self.linear2.bias, 0.)
            nn.init.constant_(self.linear3.bias, 0.)


    def forward(self, x):
        '''
        :param x: (batch * repeat_num for node/edge, emb)
        :return: w and b for affine operation
        '''
        if self.apply_batch_norm:
            x = self.bn_before(x)

        x = F.tanh(self.linear2(F.relu(self.linear1(x))))
        x = self.linear3(x)
        #x = self.rescale1(x)
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.softplus(s)
        s = self.rescale1(s) # linear scale seems important, similar to learnable prior..
        return s, t


class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            print(self.weight)
            raise RuntimeError('Rescale factor has NaN entries')

        x = torch.exp(self.weight) * x
        return x



class NetModel(nn.Module):
    def __init__(self, num_flow_layer, graph_size,
                 node_dim, args, nhid=128, nout=128):
        '''
        :param index_nod_edg:
        '''
        super(NetModel, self).__init__()
        self.graph_size = graph_size
        self.node_dim = node_dim
        self.args = args
        self.is_batchNorm = self.args.is_bn


        self.emb_size = nout
        self.hid_size = nhid
        self.num_flow_layer = num_flow_layer

        self.rgcn = RGCN(self.node_dim, nhid=self.hid_size, nout=self.emb_size,
                        num_layers=self.args.gcn_layer, dropout=self.args.dropout, normalization=False,gcn_bias=args.bias)

        if self.is_batchNorm:
            self.batchNorm = nn.BatchNorm1d(nout)

        self.st_net_fn_dict = {'sigmoid': ST_Net_Sigmoid,
                               'exp': ST_Net_Exp,
                               'softplus': ST_Net_Softplus,
                              }
        assert self.args.st_type in ['sigmoid', 'exp', 'softplus'], 'unsupported st_type, choices are [sigmoid, exp, softplus, ]'
        st_net_fn = self.st_net_fn_dict[self.args.st_type]


        #Maps from dimensions (batch,k) -> (batch, D)
        self.node_mean_net = nn.ModuleList([ST_Net_Sigmoid(self.args, self.emb_size, self.node_dim, hid_dim=nhid, bias=True,
                                                 scale_weight_norm=self.args.scale_weight_norm, sigmoid_shift=self.args.sigmoid_shift, 
                                                 apply_batch_norm=self.args.is_bn_before) for i in range(num_flow_layer)])


        #Maps from dimensions (batch,k) -> (batch,(D^2 + D) / 2)
        out_dimension = int((self.node_dim * (self.node_dim + 1)) / 2)
        self.node_cov_net = nn.ModuleList([ST_Net_Sigmoid(self.args, self.emb_size, out_dimension, hid_dim=nhid, bias=True,
                                                 scale_weight_norm=self.args.scale_weight_norm, sigmoid_shift=self.args.sigmoid_shift, 
                                                 apply_batch_norm=self.args.is_bn_before) for i in range(num_flow_layer)])


        #Maps from dimension (batch,3 * k) -> (batch)
        self.edge_mean_net = nn.ModuleList([ST_Net_Sigmoid(self.args, self.emb_size*3, 1, hid_dim=nhid, bias=True,
                                                 scale_weight_norm=self.args.scale_weight_norm, sigmoid_shift=self.args.sigmoid_shift, 
                                                 apply_batch_norm=self.args.is_bn_before) for i in range(num_flow_layer)])

        #Maps from dimensnio (batch, 3 * k) -> (batch)
        self.edge_var_net = nn.ModuleList([ST_Net_Sigmoid(self.args, self.emb_size*3, 1, hid_dim=nhid, bias=True,
                                                 scale_weight_norm=self.args.scale_weight_norm, sigmoid_shift=self.args.sigmoid_shift, 
                                                 apply_batch_norm=self.args.is_bn_before) for i in range(num_flow_layer)])

    def forward(self, x, adj, x_deq, adj_deq):
        '''
        :param x:   (batch, N, NODE_FEATURES)
        :param adj: (batch, N, N)

        :param x_deq: (batch, N, NODE_FEATURES)
        :param adj_deq:  (batch, N,N)
        :return:
        '''
        # inputs for RelGCNs
        batch_size = x.size(0)

        embedding = self.rgcn(x,adj)    #(batch,N,k)
        aggregate_embedding = torch.sum(embedding,1)    #(batch,k)


        if self.args.cuda:
            embedding = embedding.cuda()
            aggregate_embedding = aggregate_embedding.cuda()

        node_mean  = self.node_mean_net[0](aggregate_embedding)    #(batch,D)
        

        node_probs = torch.zeros_like(node_mean)
        node_probs[x[:,0,:] == 1] = node_mean[x[:,0,:] == 1]
        node_probs[x[:,0,:] == 0] = 1 - node_mean[x[:,0,:] == 0]



        '''
        x_center = x_deq[:,0,:] - node_mean

        
        L = self.node_cov_net[0](aggregate_embedding) #(batch,(D^2 + D) / 2)
        L = vector_to_Lower_Triangular(L) #(batch, D, D)


        #Asserts nonzero diagonal entries
        d = x_center.size(-1)
        epsilon =1e-4
        offset = epsilon * torch.eye(d)
        L = L + offset


        node_probs = evaluate_probability_multi(x_center,L) #(batch)
        '''




        edge_probs = torch.zeros((batch_size,self.graph_size)) #(batch, N)
        
        if self.args.cuda:
            edge_probs = edge_probs.cuda()

        for i in range(self.graph_size):
            
            edge_embedding = torch.cat((aggregate_embedding,embedding[:,0,:].clone(),embedding[:,i,:].clone()),1)


            edge_mean = self.edge_mean_net[0](edge_embedding)
            edge_std = self.edge_var_net[0](edge_embedding)

            edge_p = torch.zeros_like(edge_mean)
            a = adj[:,0,i]
            edge_p[a == 1] = edge_mean[a == 1]
            edge_p[a == 0] = 1 - edge_mean[a == 0]

            edge_probs[:,i] = torch.squeeze(edge_p)


            '''
            adj_value = adj_deq[:,0,i] - torch.squeeze(edge_mean)
            edge_probs[:,i] = evaluate_probability(adj_value,edge_std)
            '''

        #Avoids calculating log(0) terms  
        node_probs += 1e-8
        edge_probs += 1e-8

        if (node_probs == 0).any():
            print('node-zeros')
        if (edge_probs == 0).any():
            print('edge-zeros')

        node_losses = -torch.log(node_probs)
        edge_losses = -torch.log(edge_probs)

        node_losses = torch.sum(node_losses,1)
        edge_losses = torch.sum(edge_losses,1) #Reduces from (batch, N) to (batch)

        

        if self.args.cuda:
            edge_losses = edge_losses.cuda()

        return node_losses,edge_losses



