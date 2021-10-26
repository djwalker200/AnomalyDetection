#coding=utf-8
"""
Anonymous author
"""

import numpy as np
import networkx as nx

import torch
from torch.utils.data import Dataset


def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output

    
class PretrainDataset(Dataset):
    # N - Total number of graphs
    # M - Maximum number of nodes in a single graph
    # D - Number of node types (1)
    # B - Number of edge types (1)
    def __init__(self, node_features, adj_features, mol_sizes):  
        #Stores the number of graphs      
        self.n_graphs = node_features.shape[0]
        #Stores the node feature matrices for each graph (N x M x D)
        self.node_features = node_features
        #Stores the adjacency tensors for each graph (N x B x M x M)
        self.adj_features = adj_features
        #Stores the number of nodes im each graph (N x 1)
        self.mol_sizes = mol_sizes
        # Stores the maximum number of nodes for a graph (M)
        self.max_size = self.node_features.shape[1]
        #Stores the number of node types (D)
        self.node_dim = node_features.shape[2]
        #Stores the numbe of edge types (B)
        self.edge_dim = adj_features.shape[1]

    def __len__(self):
        return self.n_graphs

    def __getitem__(self, idx):
        node_feature_copy = self.node_features[idx].copy() #(N,NODE_FEATURES)
        adj_feature_copy = self.adj_features[idx].copy().astype(np.float32) #(EDGE_FEATURES, N, N)
        mol_size = self.mol_sizes[idx]

        # get permutation and bfs
        
        #Copies the node feature matrix with no changes
        node_feature = node_feature_copy
        adj_feature = adj_feature_copy

        # self connection is added for each slice.  
        for i in range(self.edge_dim):
    
            adj_feature[i] += np.eye(self.max_size)

        return {'node': torch.Tensor(node_feature), 'adj': torch.Tensor(adj_feature)}



class DataIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        
    def __next__(self):
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data