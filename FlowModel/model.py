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

from MaskGAF import MaskedGraphAF

#from rdkit import Chem
import environment as env
from environment import check_valency, convert_radical_electrons_to_hydrogens
from utils import save_one_mol, save_one_reward


class GraphFlowModel(nn.Module):
    """
    Reminder:
        self.args: deq_coeff
                   deq_type

    Args:

    
    Returns:

    """
    def __init__(self, max_size, node_dim, edge_dim, edge_unroll, args):
        super(GraphFlowModel, self).__init__()
        self.max_size = max_size
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.edge_unroll = edge_unroll
        self.args = args

        ###Flow hyper-paramters
        self.num_flow_layer = self.args.num_flow_layer
        self.nhid = self.args.nhid
        self.nout = self.args.nout

        self.node_masks = None
        if self.node_masks is None:
            self.node_masks, self.adj_masks, \
                self.link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = self.node_masks.size(0)  # (max_size) + (max_edge_unroll - 1) / 2 * max_edge_unroll + (max_size - max_edge_unroll) * max_edge_unroll
        self.latent_node_length = self.max_size * self.node_dim
        self.latent_edge_length = (self.latent_step - self.max_size) * (self.edge_dim) 
        print('latent node length: %d' % self.latent_node_length)
        print('latent edge length: %d' % self.latent_edge_length)

        self.constant_pi = nn.Parameter(torch.Tensor([3.1415926535]), requires_grad=False)
        #learnable
        if self.args.learn_prior:
            self.prior_ln_var = nn.Parameter(torch.zeros([1])) # log(1^2) = 0
            nn.init.constant_(self.prior_ln_var, 0.0)            
        else:
            self.prior_ln_var = nn.Parameter(torch.zeros([1]), requires_grad=False)

        self.dp = False
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 1:
            self.dp = True
            print('using %d GPUs' % num_gpus)
        
        self.flow_core = MaskedGraphAF(self.node_masks, self.adj_masks, 
                                       self.link_prediction_index, 
                                       num_flow_layer = self.num_flow_layer,
                                       graph_size=self.max_size,
                                       num_node_type=self.node_dim,
                                       num_edge_type=self.edge_dim,
                                       args=self.args,
                                       nhid=self.nhid,
                                       nout=self.nout)
        if self.dp:
            self.flow_core = nn.DataParallel(self.flow_core)
        


    def forward(self, inp_node_features, inp_adj_features):
        """
        Args:
            inp_node_features: (B, N, NODE_FEATURES)
            inp_adj_features: (B, EDGE_FEATURES + 1, N, N)

        Returns:
            z: [(B, node_num* NODE_FEATURES), (B, edge_num * (EDGE_FEATURES + 1)]
            logdet:  ([B], [B])        
        """
        #TODO: add dropout/normalize


        #inp_node_features_cont = inp_node_features #(B, N, NODE_FEATURES) #! this is buggy. shallow copy
        inp_node_features_cont = inp_node_features.clone() #(B, N, NODE_FEATURES)

        inp_adj_features_cont = inp_adj_features[:,:, self.flow_core_edge_masks].clone() #(B, EDGE_FEATURES, edge_num)
        inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous() #(B, edge_num, EDGE_FEATURES)

        if self.args.deq_type == 'random':
            #TODO: put the randomness on GPU.!
            #CUDA MODIFICATION
            if self.args.cuda:
                inp_node_features_cont += self.args.deq_coeff * torch.rand(inp_node_features_cont.size()).cuda() #(BATCH_SIZE, NUM_NODES, NODE_FEATURES)
                inp_adj_features_cont += self.args.deq_coeff * torch.rand(inp_adj_features_cont.size()).cuda() #(BATCH_SIZE, edge_num, EDGE_FEATURES)
            else:
                inp_node_features_cont += self.args.deq_coeff * torch.rand(inp_node_features_cont.size()) #(BATCH_SIZE, NUM_NODES, NODE_FEATURES)
                inp_adj_features_cont += self.args.deq_coeff * torch.rand(inp_adj_features_cont.size()) #(BATCH_SIZE, edge_num, EDGE_FEATURES)

        elif self.args.deq_type == 'variational':
            #TODO: add variational deq.
            raise ValueError('current unsupported method: %s' % self.args.deq_type)
        else:
            raise ValueError('unsupported dequantization type (%s)' % self.args.deq_type)


        z, logdet = self.flow_core(inp_node_features, inp_adj_features, 
                                   inp_node_features_cont, inp_adj_features_cont)
        
        if self.args.deq_type == 'random':
            return z, logdet, self.prior_ln_var

        elif self.args.deq_type == 'variational':
            #TODO: try variational dequantization
            return z, logdet, deq_logp, deq_logdet


 
    def generate(self, temperature=0.75, mute=False, max_atoms=48, cnt=None):
        """
        inverse flow to generate molecule
        Args: 
            temp: temperature of normal distributions, we sample from (0, temp^2 * I)
        """
        generate_start_t = time()
        with torch.no_grad():
            '''
            num2bond =  {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
            num2bond_symbol =  {0: '=', 1: '==', 2: '==='}
            # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4:15, 5:16, 6:17, 7:35, 8:53}
            num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4:'P', 5:'S', 6:'Cl', 7:'Br', 8:'I'}

            '''
            if self.args.cuda:

                prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(), 
                                            temperature * torch.ones([self.node_dim]).cuda())
                prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.edge_dim + 1]).cuda(), 
                                            temperature * torch.ones([self.edge_dim + 1]).cuda())

                cur_node_features = torch.zeros([1, max_atoms, self.node_dim]).cuda()
                cur_adj_features = torch.zeros([1, self.edge_dim + 1, max_atoms, max_atoms]).cuda()
            else:
                prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]), 
                                            temperature * torch.ones([self.node_dim]))
                prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.edge_dim + 1]), 
                                            temperature * torch.ones([self.edge_dim + 1]))   

                cur_node_features = torch.zeros([1, max_atoms, self.node_dim])
                cur_adj_features = torch.zeros([1, self.edge_dim + 1, max_atoms, max_atoms])

            #rw_mol = Chem.RWMol() # editable mol
            mol = None
            #mol_size = mol.GetNumAtoms()

            is_continue = True
            total_resample = 0
            each_node_resample = np.zeros([max_atoms])
            for i in range(max_atoms):
                if not is_continue:
                    break
                #Sets the number of edges to consider, and the starting index to check from
                if i < self.edge_unroll:
                    edge_total = i # edge to sample for current node
                    start = 0
                else:
                    edge_total = self.edge_unroll
                    start = i - self.edge_unroll

                # first generate node
                ## reverse flow
                latent_node = prior_node_dist.sample().view(1, -1) #(1, NODE_FEATURES)
                
                #latent_node stores a dequantized one-hot vector encoding the features
                if self.dp:
                    latent_node = self.flow_core.module.reverse(cur_node_features, cur_adj_features, 
                                            latent_node, mode=0).view(-1) # (NODE_FEATURES, )
                else:

                    latent_node = self.flow_core.reverse(cur_node_features, cur_adj_features, 
                                            latent_node, mode=0).view(-1) # (NODE_FEATURES, )
                ## node/adj postprocessing
                #print(latent_node.shape) #(MAX_SIZE, NODE_FEATURES)
                #Finds the feature_id (atomic number) from the one-hot encoding
                feature_id = torch.argmax(latent_node).item()
                #print(num2symbol[feature_id])
                cur_node_features[0, i, feature_id] = 1.0
                cur_adj_features[0, :, i, i] = 1.0
                #rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))
                

                # then generate edges
                if i == 0:
                    is_connect = True
                else:
                    is_connect = False
                #cur_mol_size = mol.GetNumAtoms
                for j in range(edge_total):
                    valid = False
                    resample_edge = 0
                    invalid_bond_type_set = set()

                    while not valid:
                        #TODO: add cache. Some atom can not get the right edge type and is stuck in the loop
                        #TODO: add cache. invalid bond set

                        if len(invalid_bond_type_set) < self.edge_dim and resample_edge <= 50: # haven't sampled all possible bond type or is not stuck in the loop
                            latent_edge = prior_edge_dist.sample().view(1, -1) #(1, 4)

                            if self.dp:

                                latent_edge = self.flow_core.module.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1) #(4, )
                            else:

                                latent_edge = self.flow_core.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[j + start, i]]).long()).view(-1) #(4, )

                            edge_discrete_id = torch.argmax(latent_edge).item()
                        else:

                            if not mute:
                                print('have tried all possible bond type, use virtual bond.')

                            assert resample_edge > 50 or len(invalid_bond_type_set) == self.edge_dim
                            edge_discrete_id = self.edge_dim # (3) we have no choice but to choose not to add edge between (i, j+start)

                        cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                        cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0


                        if edge_discrete_id == self.edge_dim: # virtual edge 
                            valid = True
                        else: #single/double/triple bond
                            #rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])                                                   
                            #valid = env.check_valency(rw_mol)
                            valid = True
                            if valid:
                                is_connect = True
                                #print(num2bond_symbol[edge_discrete_id])
                            else: #backtrack
                                rw_mol.RemoveBond(i, j + start)
                                cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                total_resample += 1.0
                                each_node_resample[i] += 1.0
                                resample_edge += 1
                                #If the edge type is not allowed between those two atoms, store in the invalid set
                                invalid_bond_type_set.add(edge_discrete_id)


                num_atoms = i + 1
                if is_connect: # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                    is_continue = True
                    #mol = rw_mol.GetMol()
                
                else:
                    is_continue = False

            '''
            mol = rw_mol.GetMol() # mol backup
            assert mol is not None, 'mol is None...'

            final_valid = check_valency(mol)
            final_valid = env.check_chemical_validity(mol)            
            assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!'            

            final_mol = env.convert_radical_electrons_to_hydrogens(mol)
            smiles = Chem.MolToSmiles(final_mol, isomericSmiles=True)
            assert '.' not in smiles, 'warning: use is_connect to check stop action, but the final molecule is disconnected!!!'

            final_mol = Chem.MolFromSmiles(smiles)


            mol = convert_radical_electrons_to_hydrogens(mol)
            num_atoms = final_mol.GetNumAtoms()
            num_bonds = final_mol.GetNumBonds()
            '''

            pure_valid = 0
            if total_resample == 0:
                pure_valid = 1.0
            if not mute:
                cnt = str(cnt) if cnt is not None else ''
                print('smiles%s: %s | #atoms: %d | #bonds: %d | #resample: %.5f | time: %.5f |' % (cnt, smiles, num_atoms, num_bonds, total_resample, time()-generate_start_t))
            return cur_node_features,cur_adj_features, pure_valid, num_atoms
    

    def initialize_masks(self, max_node_unroll=38, max_edge_unroll=12):
        """
        Args:
            max node unroll: maximal number of nodes in molecules to be generated (default: 38)
            max edge unroll: maximal number of edges to predict for each generated nodes (default: 12, calculated from zink250K data)
        Returns:
            node_masks: node mask for each step
            adj_masks: adjacency mask for each step
            is_node_update_mask: 1 indicate this step is for updating node features
            flow_core_edge_mask: get the distributions we want to model in adjacency matrix
        """
        
        #MY COMMENTS
        
        #N = maximum nodes in a graph
        #P = maximum edge dependency from BFS

        #Total number of masks that will be created
        num_masks = int(max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (max_node_unroll - max_edge_unroll) * (max_edge_unroll))
        #Maximal number of potential edges in the graph
        num_mask_edge = int(num_masks - max_node_unroll)

        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).byte() #(N , N)
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).byte() #(N , N, N)
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).byte() #(num_masks - P,N)
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).byte()    #(num_masks - P,N,N)     
        #is_node_update_masks = torch.zeros([num_masks]).byte()

        #Stores the indices for every potential pair of edges (u,v)
        link_prediction_index = torch.zeros([num_mask_edge, 2]).long() #(num_masks - P, 2)
        '''
        Looks like this:
        [[0,1],
        [0,2],
        [1,2],
        ...,
        [N,N]]
        '''

        #Mask that stores all potential edge pairings, given the BFS ordering and maximum dependency P
        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).byte() #(N,N)
        '''
        Looks like this (example P = 4):
        [[0 0 0 .... 0 0],
         [1 0 0 .... 0 0],
         [1 1 0 .... 0 0],
         ...
         [0 0 ... 0 1 1 1 1 0 ....  0 0],
         ...
         [0 0 .... 0 1 1 1 1]]

        where a 1 in position (r,c) denotes that node c could have an edge to node r given the dependency
        '''


        cnt = 0
        cnt_node = 0
        cnt_edge = 0
        #Loops over every node, up to the maximum graph size
        for i in range(max_node_unroll):
            #Places 1s in position corresponding to all nodes preceeding the current node in BFS ordering
            node_masks1[cnt_node][:i] = 1
            adj_masks1[cnt_node][:i, :i] = 1
            #is_node_update_masks[cnt] = 1
            #Increments counts
            cnt += 1
            cnt_node += 1
            #Total number of edges to consider, limited by the BFS dependency limit 
            edge_total = 0
            #If checking the first 1->max_edge_unroll nodes, all potential edges must be checked
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                #Else, only the maximum dependency number of edges must be checked
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
            #Loops over all potential edges to the current node
            for j in range(edge_total):
                if j == 0:
                    node_masks2[cnt_edge][:i+1] = 1
                    adj_masks2[cnt_edge] = adj_masks1[cnt_node-1].clone()
                    adj_masks2[cnt_edge][i,i] = 1
                else:
                    node_masks2[cnt_edge][:i+1] = 1
                    adj_masks2[cnt_edge] = adj_masks2[cnt_edge-1].clone()
                    adj_masks2[cnt_edge][i, start + j -1] = 1
                    adj_masks2[cnt_edge][start + j -1, i] = 1
                cnt += 1
                cnt_edge += 1


        #Asserts that all edges and nodes are accounted for
        assert cnt == num_masks, 'masks cnt wrong'
        assert cnt_node == max_node_unroll, 'node masks cnt wrong'
        assert cnt_edge == num_mask_edge, 'edge masks cnt wrong'

    
        cnt = 0
        #Loops over every node, up to the maximum graph size
        for i in range(max_node_unroll):
            #If checking the first 1->max_edge_unroll nodes, all potential edges must be checked
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                #Else, only the maximum dependency number of edges must be checked
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
        
            #Loops over all potential edges for the current node
            for j in range(edge_total):
                link_prediction_index[cnt][0] = start + j
                link_prediction_index[cnt][1] = i
                cnt += 1

        #Asserts that all potential edges were accounted for
        assert cnt == num_mask_edge, 'edge mask initialize fail'

        #Loops over every node, up to the maximum graph size
        for i in range(max_node_unroll):
            if i == 0:
                continue
            #If checking the first 1->max_edge_unroll nodes, all potential edges must be checked
            if i < max_edge_unroll:
                start = 0
                end = i
            else:
                #Else, only the maximum dependency number of edges must be checked
                start = i - max_edge_unroll
                end = i 

            #Places 1s in column of all preceeding nodes that could have an edge to node i
            flow_core_edge_masks[i][start:end] = 1

        #Forms the node and adjacency masks
        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        node_masks = nn.Parameter(node_masks, requires_grad=False)
        adj_masks = nn.Parameter(adj_masks, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)
        
        return node_masks, adj_masks, link_prediction_index, flow_core_edge_masks




    def log_prob(self, z, logdet, deq_logp=None, deq_logdet=None,aggregate_loss = True):
          

        #TODO: check multivariate gaussian log_prob formula
        #logdet[0] = logdet[0] - self.latent_node_length # calculate probability of a region from probability density, minus constant has no effect on optimization
        #logdet[1] = logdet[1] - self.latent_edge_length # calculate probability of a region from probability density, minus constant has no effect on optimization

        

        ll_node = -1/2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[0]**2))
        ll_node = ll_node.sum(-1) # (B)
        ll_edge = -1/2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[1]**2))
        ll_edge = ll_edge.sum(-1) # (B)

        ll_node += logdet[0] #([B])
        ll_edge += logdet[1] #([B])


        if self.args.deq_type == 'random':
            if self.args.divide_loss:
                if aggregate_loss:
                    return -(torch.mean(ll_node + ll_edge) / (self.latent_edge_length + self.latent_node_length))
                else:
                    return -(ll_node + ll_edge) / (self.latent_edge_length + self.latent_node_length)
            else:
                #! useless
                if aggregate_loss:
                    return -torch.mean(ll_node + ll_edge) # scalar
                else:
                    return -(ll_node + ll_edge)

        elif self.args.deq_type == 'variational':
            #TODO: finish this part
            assert deq_logp is not None and deq_logdet is not None, 'variational dequantization requires deq_logp/deq_logdet'
            ll_deq_node = deq_logp[0] - deq_logdet[0] #()
            #print(ll_deq_node.size())
            ll_deq_edge = deq_logp[1] - deq_logdet[1]
            #print(ll_deq_edge.size())
            return (torch.mean(ll_node), torch.mean(ll_edge), torch.mean(ll_deq_node), torch.mean(ll_deq_edge))

        
        

