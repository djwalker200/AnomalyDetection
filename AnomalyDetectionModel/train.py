 #coding: utf-8
'''
Anonymous author
'''

from time import time
import argparse
import numpy as np
import math
import os
import sys
import json


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import *

from model import GraphFlowModel
from dataloader import PretrainDataset


from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def save_model(model, optimizer, args, var_list, epoch=None):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(argparse_dict, f)

    epoch = str(epoch) if epoch is not None else ''
    latest_save_path = os.path.join(args.save_path, 'checkpoint')
    final_save_path = os.path.join(args.save_path, 'checkpoint%s' % epoch)
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        final_save_path
    )

    # save twice to maintain a latest checkpoint
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        latest_save_path
    )    


def restore_model(model, args, epoch=None):
    if epoch is None:
        restore_path = os.path.join(args.save_path, 'checkpoint')
        print('restore from the latest checkpoint')
    else:
        restore_path = os.path.join(args.save_path, 'checkpoint%s' % str(epoch))
        print('restore from checkpoint%s' % str(epoch))

    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['model_state_dict'])


def read_graphs(folder,name):
    path = os.getcwd() + '/' #+ folder + '/'
    node_features = np.load(path + name + '_node_features.npy')
    adj_features = np.load(path + name +  '_adj_features.npy')
    mol_sizes = np.load(path + name + '_sizes.npy')


    return node_features, adj_features, mol_sizes

    


class Trainer(object):
    def __init__(self,dataloader, validation_dataloader, data_config, args):
        self.dataloader = dataloader
        self.valid_dataloader = validation_dataloader
        self.data_config = data_config
        self.args = args
        

        self.max_size = self.data_config['max_size']
        self.node_dim = self.data_config['node_dim'] 
       
        
        self._model = GraphFlowModel(self.max_size, self.node_dim, self.args)
        self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()),
                        lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.best_loss = 100.0
        self.start_epoch = 0
        if self.args.cuda:
            self._model = self._model.cuda()
    

    def initialize_from_checkpoint(self, gen=False):
        checkpoint = torch.load(self.args.init_checkpoint)
        self._model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if not gen:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_loss = checkpoint['best_loss']
            self.start_epoch = checkpoint['cur_epoch'] + 1
        print('initialize from %s Done!' % self.args.init_checkpoint)


    def fit(self, mol_out_dir=None):        
        t_total = time()
        total_loss = []
        best_loss = self.best_loss
        start_epoch = self.start_epoch

        print('start fitting.')
        for epoch in range(self.args.epochs):
            epoch_loss = self.train_epoch(epoch + start_epoch)
            total_loss.append(epoch_loss)
            

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if self.args.save:
                    var_list = {'cur_epoch': epoch + start_epoch,
                                'best_loss': best_loss,

                               }
                    save_model(self._model, self._optimizer, self.args, var_list, epoch=epoch + start_epoch)
        

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time() - t_total))            

        return total_loss


    def train_epoch(self, epoch_cnt):

        t_start = time()
        batch_losses = []
        self._model.train()
        batch_cnt = 0
        
        for i_batch, batch_data in enumerate(self.dataloader):
            batch_time_s = time()

            self._optimizer.zero_grad()

            batch_cnt += 1
            inp_node_features = batch_data['node'] #(BATCH_SIZE, NUM_NODES, NODE_FEATURES)
            inp_adj_features = batch_data['adj'] #(BATCH_SIZE, NUM_NODES, NUM_NODES)   


            if self.args.cuda:
                inp_node_features = inp_node_features.cuda()
                inp_adj_features = inp_adj_features.cuda()

            if self.args.deq_type == 'random':
                l_nodes, l_edges = self._model(inp_node_features, inp_adj_features)
                loss =  torch.mean(l_nodes + l_edges)


                #TODO: add mask for different molecule size, i.e. do not model the distribution over padding nodes.

            elif self.args.deq_type == 'variational':
                out_z, out_logdet, out_deq_logp, out_deq_logdet = self._model(inp_node_features, inp_adj_features)
                ll_node, ll_edge, ll_deq_node, ll_deq_edge = self._model.log_prob(out_z, out_logdet, out_deq_logp, out_deq_logdet)
                loss = -1. * ((ll_node-ll_deq_node) + (ll_edge-ll_deq_edge))
            else:
                raise ValueError('unsupported dequantization method: (%s)' % self.deq_type)

            loss.backward()
            self._optimizer.step()

            batch_losses.append(loss.item())

        #Find validation loss
        valid_losses = []
        for i_batch,batch_data in enumerate(self.valid_dataloader):

            self._optimizer.zero_grad()

            inp_node_features = batch_data['node'] #(BATCH_SIZE, NUM_NODES, NODE_FEATURES)
            inp_adj_features = batch_data['adj'] #(BATCH_SIZE, EDGE_FEATURES + 1, NUM_NODES, NUM_NODES)   


            if self.args.cuda:
                inp_node_features = inp_node_features.cuda()
                inp_adj_features = inp_adj_features.cuda()

            if self.args.deq_type == 'random':
                l_nodes,l_edges = self._model(inp_node_features, inp_adj_features)
                loss = torch.mean(l_nodes + l_edges)

            valid_losses.append(loss.item())
        
        valid_loss = sum(valid_losses) / len(valid_losses)
        epoch_loss = sum(batch_losses) / len(batch_losses)

        if epoch_cnt % 5 == 0:
            print('Epoch: {: d}, loss {:5.5f}, epoch time {:.5f}'.format(epoch_cnt, epoch_loss, time()-t_start))  

        return valid_loss

#Finds the log-probabilities for an entire dataset
def find_distribution(dataloader,model,args):


    for i_batch,batch_data in enumerate(dataloader):

        #Reads in the batch adjacency tensors and node feature matrices
        adj = batch_data['adj']
        nodes = batch_data['node']

        if args.cuda:
            nodes = nodes.cuda()
            adj = adj.cuda()

        if args.deq_type == 'random':
            #Runs the model on the test batch
            l_nodes,l_edges = model(nodes,adj)
            loss = l_nodes + l_edges

        #Updates the array of log-likelihoods
        if i_batch == 0:
            probs = np.array([loss.detach().cpu().numpy()])
        else:
            l = np.array([loss.detach().cpu().numpy()])
            probs = np.concatenate((probs,l),axis=1)

    #Returns the log-likelihood for each test graph
    return -1 * probs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphFlow model')

    # ******data args******
    parser.add_argument('--dataset', type=str, default='zinc250k', help='dataset')
    parser.add_argument('--path', type=str, help='path of dataset', required=True)


    parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle data for each epoch')
    parser.add_argument('--num_workers', type=int, default=3, help='num works to generate data.')

    # ******model args******
    parser.add_argument('--name', type=str, default='base', help='model name, crucial for test and checkpoint initialization')
    parser.add_argument('--deq_type', type=str, default='random', help='dequantization methods.')
    parser.add_argument('--deq_coeff', type=float, default=0.0, help='dequantization coefficient.(only for deq_type random)')
    parser.add_argument('--num_flow_layer', type=int, default=1, help='num of affine transformation layer in each timestep')
    parser.add_argument('--gcn_layer', type=int, default=3, help='num of rgcn layers')
    #TODO: Disentangle num of hidden units for gcn layer, st net layer.
    parser.add_argument('--nhid', type=int, default=128, help='num of hidden units of gcn')
    parser.add_argument('--nout', type=int, default=128, help='num of out units of gcn')

    parser.add_argument('--st_type', type=str, default='sigmoid', help='architecture of st net, choice: [sigmoid, exp, softplus, spine]')

    # ******for sigmoid st net only ******
    parser.add_argument('--sigmoid_shift', type=float, default=2.0, help='sigmoid shift on s.')

    # ******for exp st net only ******

    # ******for softplus st net only ******

    # ******optimization args******
    parser.add_argument('--all_save_prefix', type=str, default='./', help='path of save prefix')
    parser.add_argument('--train', action='store_true', default=False, help='do training.')
    parser.add_argument('--save', action='store_true', default=False, help='Save model.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--learn_prior', action='store_true', default=False, help='learn log-var of gaussian prior.')

    parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--is_bn', action='store_true', default=False, help='batch norm on node embeddings.')
    parser.add_argument('--is_bn_before', action='store_true', default=False, help='batch norm on node embeddings on st-net input.')
    parser.add_argument('--scale_weight_norm', action='store_true', default=False, help='apply weight norm on scale factor.')
    parser.add_argument('--divide_loss', action='store_true', default=False, help='divide loss by length of latent.')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='initialize from a checkpoint, if None, do not restore')

    parser.add_argument('--show_loss_step', type=int, default=100)
    parser.add_argument('--bias',action='store_true',default=False,help='Bias for rgcn layers')
    # ******generation args******
    parser.add_argument('--temperature', type=float, default=0.75, help='temperature for normal distribution')
    parser.add_argument('--min_atoms', type=int, default=5, help='minimum #atoms of generated mol, otherwise the mol is simply discarded')
    parser.add_argument('--max_atoms', type=int, default=15, help='maximum #atoms of generated mol')    
    parser.add_argument('--gen_num', type=int, default=100, help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--gen', action='store_true', default=False, help='generate')
    parser.add_argument('--gen_out_path', type=str, help='output path for generated mol')

    



    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.save:
        checkpoint_dir = args.all_save_prefix + 'save_pretrain/%s_%s_%s' % (args.st_type, args.dataset, args.name)
        args.save_path = checkpoint_dir

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    set_seed(args.seed, args.cuda)



    assert (args.train and not args.gen) or (args.gen and not args.train), 'please specify either train or gen mode'

    #Reads in the data configurations
    path = os.getcwd() + '/'
    f = open(path + '_config.txt', 'r')
    data_config = eval(f.read())
    f.close()

    #Reads in the data from the training network
    train_node_features, train_adj_features, train_sizes = read_graphs(args.path,'train')
    valid_node_features, valid_adj_features, valid_sizes = read_graphs(args.path,'valid')
    test_node_features,test_adj_features,test_sizes  = read_graphs(args.path,'test')

    path = os.getcwd() + '/' #+ folder + '/'
    train_anoms = np.load(path +  'anoms.npy')


    #Creates a dataloader for each data subset
    train_dataloader = DataLoader(PretrainDataset(train_node_features,train_adj_features,train_sizes),
                                batch_size=args.batch_size,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
    valid_dataloader = DataLoader(PretrainDataset(valid_node_features,valid_adj_features,valid_sizes),
                                batch_size=args.batch_size,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
    test_dataloader = DataLoader(PretrainDataset(test_node_features,test_adj_features,test_sizes),
                                batch_size= args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers)
    
    #Initializes the model trainer and trains the model
    trainer = Trainer(train_dataloader,valid_dataloader, data_config, args)


    if args.init_checkpoint is not None:
        trainer.initialize_from_checkpoint(gen=args.gen)
    if args.train:
        if args.save:
            mol_out_dir = os.path.join(checkpoint_dir, 'mols')

            if not os.path.exists(mol_out_dir):
                os.makedirs(mol_out_dir)
        else:
            mol_out_dir = None

        total_loss = trainer.fit(mol_out_dir=mol_out_dir)

    #Creates a plot of the losses as a function of the epoch
    plt.plot(total_loss)
    plt.title("Validation Epoch Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('epoch_losses.png')
    plt.clf()


    #Gathers the trained model for testing
    test_model = trainer._model
    test_model.eval()

    #Loads anomaly data
    anom_node_features,anom_adj_features,anom_sizes = read_graphs(args.path,'extra')
    anom_dataloader = DataLoader(PretrainDataset(anom_node_features,anom_adj_features,anom_sizes),
                                batch_size=args.batch_size,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
        
    #Calculates the log-prob distribution for each set

    train_node_features = train_node_features[train_anoms == 0]
    train_adj_features = train_adj_features[train_anoms == 0]
    train_sizes = train_sizes[train_anoms == 0]



    train_dataloader = DataLoader(PretrainDataset(train_node_features,train_adj_features,train_sizes),
                                batch_size=args.batch_size,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
    train_probs = find_distribution(train_dataloader,test_model,args).reshape(-1)
    test_probs = find_distribution(test_dataloader,test_model,args).reshape(-1)
    anom_probs = find_distribution(anom_dataloader,test_model,args).reshape(-1)



    #Concatenates
    test_probs = np.concatenate((train_probs,test_probs))
    total_probs = np.concatenate((test_probs,anom_probs))

    #Sets limits for graphics

    min_prob = np.min(total_probs)
    max_prob = np.max(total_probs)
    min_x = min_prob - 0.25 * (max_prob - min_prob)
    max_x = max_prob + 0.25 * (max_prob - min_prob)

    #Plots histograms for each set with a kde plot overlay
    #sns.histplot(data=train_probs,color='b',stat="density",label='True', kde=True)
    sns.histplot(data=test_probs,color='g',stat="density",label='True',kde=True)
    sns.histplot(data=anom_probs,color='r',stat="density",label='Anomalies',kde=True)
    plt.xlim(min_x,max_x)
    plt.legend()
    plt.xlabel('Log-likelihood')
    plt.savefig('prob_distribution.png')
    plt.clf()

    # Forms ROC curve and calculates AUC score
    # Takes only the testing data and anomaly data (excludes training data)
    Y_probs = total_probs
    # Initializes class label 1 for all graphs from the test set and 0 for all graphs from the anomaly dataset
    Y_true = np.zeros(Y_probs.shape[0])
    Y_true[test_probs.shape[0]:] = 1
    #Calculates the ROC and AUC statistics
    fpr,tpr,thresholds = roc_curve(Y_true,-1 * Y_probs)
    roc_auc = auc(fpr,tpr)

    #Finds the optimal threshold
    geometric_means = np.sqrt(tpr * (1 - fpr))
    best_threshold_index = np.argmax(geometric_means)
    best_threshold = - thresholds[best_threshold_index] + 1e-5
    print('Best Threshold=%f:' % best_threshold)

    #Plots ROC Curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.scatter(fpr[best_threshold_index], tpr[best_threshold_index], marker='o', color='black', label='Best Threshold')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('ROC_curve.png')


    Y_pred = np.zeros(Y_true.shape[0])
    Y_pred[Y_probs < best_threshold] = 1
    f1 = f1_score(Y_true,Y_pred)

    print('AUC score: %.3f' % roc_auc)
    print('F1 score: %.3f' % f1)

    results = []
    for i in range(Y_probs.shape[0]):
        results.append((Y_probs[i],Y_true[i]))

    results.sort(key = lambda x: x[0])

    n_anomalies = anom_probs.shape[0]
    
    for k in [50,100]:
        n_detected = sum((pred < best_threshold) for (pred,__) in results[:k])
        n_correct = sum( (truth == 1 and pred < best_threshold)  for (pred,truth) in results[:k]) 
        if n_detected > 0:
            k_precision = float(n_correct) / n_detected
        else:
            k_precision = 0
        k_recall = float(n_correct) / n_anomalies
        print('Precision@{:.0f}: {:.3f}'.format(k,k_precision))
        print('Recall@{:.0f}: {:.3f}'.format(k,k_recall))

    detections = Y_true[Y_pred == 1]
    n_correct = detections[detections == 1].shape[0]
    if(detections.shape[0] > 0):
        precision = float(n_correct) / detections.shape[0]
    else:
        precision = 0
    recall = float(n_correct) / n_anomalies
    print('Precision:', '%.3f' % precision)
    print('Recall:','%.3f' % recall)

    if args.gen:
        print('start generating...')
        trainer.generate_molecule(num=args.gen_num, out_path=args.gen_out_path, mute=False)
