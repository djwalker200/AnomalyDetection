import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import snap
import math
import random
import scipy.io
import scipy.sparse as sp
from random import sample

import os
import sys
import json

from time import time
from input_data import generateFeatures,setupFeatures,findCenters,findNeighborhoodFullRandom, \
ordering_BFS,getLabels,neighborhood_BFS,AddCliques,Scramble,GenerateSwapAnomalies,FeatureSwapAnomalies

if __name__ == "__main__":  

    #Takes in the folder name to read data from and changes directory
    folder = sys.argv[1]

    os.chdir(os.getcwd() + '/' + folder)

    #Parameters for generating the training set

    #Number of edge features
    EDGE_FEATURES = 60
    #Number of node features
    NODE_FEATURES = 512
    #Number of training graphs to be generated
    n_train = int(sys.argv[2])
    #Number of validationgraph to be generated
    n_valid = int(sys.argv[3])
    #Number of testing graphs to be generated
    n_test = int(sys.argv[4])
    #Number of anomalies to generate
    n_anomalies = int(sys.argv[5])
    #Method for generating the neighborhoods
    generation_type = 'BFS'
    #Type of anomaly to generate
    anomaly_type = 'Clique'
    #Search depth for BFS
    BFS_DEPTH = 2
    #Length of walks to be used in random walk neighborhood generation
    WALK_LENGTH = 2
    #Number of walks to be used in random walk neighborhood generation
    NUM_WALKS = 60
    #Maximum allowed size for a BFS neighborhood
    MAX_SIZE = 30

    if folder == "Data":
            G = nx.read_edgelist("nell-new.csv",create_using=nx.Graph(),nodetype=str,
            delimiter=",",data=(("weight",int),))
    else:
        #Reads in an undirected networkx graph from an edge list .txt file
        G = nx.read_edgelist(folder + '_edges.txt',create_using=nx.Graph(),nodetype=str)
    cols = list(range(0,NODE_FEATURES + 1))
    cols.append(-1)
    node_feature_dict = pd.read_csv("nell-node-feature.csv", header=None, index_col=0,sep='\t', squeeze=True).to_dict()

    '''
    if NODE_FEATURES > 1:
        node_feature_matrix = np.loadtxt(folder + '_features.txt',usecols=cols,dtype=str)
        node_feature_dict,class_dict = setupFeatures(node_feature_matrix)       
    else:
        node_feature_matrix = None
        node_feature_dict = None
        class_dict = None
    '''
    print(nx.info(G))

    #Revert working directory

    os.chdir(os.getcwd()[:-len(folder)])

    #Stores the anomaly set of neighborhood adjacency matrices
    anom_matrices = np.zeros((n_anomalies,EDGE_FEATURES, MAX_SIZE,MAX_SIZE))
    #Stores the anomaly set node feature matrix
    anom_features = np.zeros((n_anomalies,MAX_SIZE,NODE_FEATURES))


    #Stores the training set 
    train_matrices = np.zeros((n_train,EDGE_FEATURES, MAX_SIZE,MAX_SIZE))
    train_features = np.zeros((n_train,MAX_SIZE,NODE_FEATURES))

    #Stores the validation set
    valid_matrices = np.zeros((n_valid,EDGE_FEATURES, MAX_SIZE,MAX_SIZE))
    valid_features = np.zeros((n_valid,MAX_SIZE,NODE_FEATURES))

    #Stores the test set 
    test_matrices = np.zeros((n_test,EDGE_FEATURES, MAX_SIZE,MAX_SIZE))
    test_features = np.zeros((n_test,MAX_SIZE,NODE_FEATURES))


    #Stores the size of each graph
    train_sizes = np.zeros(n_train).astype(np.uint8)
    valid_sizes = np.zeros(n_valid).astype(np.uint8)
    test_sizes = np.zeros(n_test).astype(np.uint8)
    anom_sizes = np.zeros(n_anomalies).astype(np.uint8)

    #Stores hyperstatistics for the graph sets
    total_triangles = []
    max_dependency = 0
    total_degrees = []

    # Extra variables to store the test data node IDs and classes (optional)
    #test_ids = []
    #test_classes = np.zeros((n_test,MAX_SIZE),dtype=np.object_)


    #Generates a set of random node indices from G
    random_nodes = sample(list(G.nodes()), n_train + n_valid + n_test)


    #Loops over each randomly selected node to generate a training neighborhood
    for i,start_node in enumerate(random_nodes):
        
        if generation_type == 'random_walk':
            SubG = findNeighborhoodFullRandom(G,start_node,WALK_LENGTH,NUM_WALKS,MAX_SIZE)
            center = findCenters(SubG)
            SubG,bfs_order,dependency = ordering_BFS(SubG,center)
        elif generation_type == 'BFS':
            SubG,bfs_order,dependency = neighborhood_BFS(G,start_node,BFS_DEPTH,MAX_SIZE)
        else:
            print('Invalid generation type')
            exit()

        max_dependency = max(max_dependency,dependency)
        #Calculates the number of triangles for the sample graph (training set only)
        if i < n_train:
            num_triangles = sum(nx.triangles(SubG).values()) / 3
            total_triangles.append(num_triangles)

        #Generates the features of the graph
        adjacency_tensor,X = generateFeatures(SubG,bfs_order,node_feature_dict,EDGE_FEATURES,NODE_FEATURES)
        
        num_nodes = X.shape[0]
        #Stores the features to the correct sets
        if  i < n_train:
            train_sizes[i] = num_nodes
            train_matrices[i,:,:num_nodes,:num_nodes] = adjacency_tensor
            train_features[i,:num_nodes] = X
            total_degrees.append(float(2 * SubG.number_of_edges()) / num_nodes)

        elif i < n_train + n_valid:
            valid_sizes[i - n_train] = num_nodes
            valid_matrices[i - n_train,:,:num_nodes,:num_nodes] = adjacency_tensor
            valid_features[i - n_train,:num_nodes] = X

        else:
            test_sizes[i - (n_train + n_valid)] = num_nodes
            test_matrices[i - (n_train + n_valid),:,:num_nodes,:num_nodes] = adjacency_tensor
            test_features[i - (n_train + n_valid),:num_nodes] = X
            #Stores the node ID that generated the current sample
            #test_ids.append(start_node)
            #Stores the class label for each node in the sample (ordered)
            #labels = getLabels(bfs_order,class_dict)
            #test_classes[i - (n_train + n_valid),:num_nodes] = labels
    




 

    


    anom_degrees = []
    anom_triangles = []

    if anomaly_type == 'Clique':
        G,anomaly_indices = AddCliques(G,num_cliques=15,clique_size=15)
        anomaly_nodes = anomaly_indices[:n_anomalies]

    elif anomaly_type == 'Structure-Swap':
        folder2 = sys.argv[5]
        H = nx.read_edgelist(folder2 + '_edges.txt',create_using=nx.Graph(),nodetype=str)
        extra_matrices,extra_features,extra_sizes,extra_degrees,extra_triangles = \
        GenerateSwapAnomalies(G,H,n_anomalies,MAX_SIZE,NODE_FEATURES,node_feature_dict)

    elif anomaly_type == 'Feature-Swap':
        anomaly_nodes = sample(list(G.nodes()),n_anomalies)
        feature_swaps = FeatureSwapAnomalies(G,anomaly_nodes,NODE_FEATURES,class_dict,node_feature_dict)

    elif anomaly_type == 'Scramble':
        node_feature_dict,anomaly_nodes = Scramble(G,node_feature_dict,n_anomalies,NODE_FEATURES)

    
    for i,start_node in enumerate(anomaly_nodes):

        if anomaly_type == 'Structure-Swap':
            break

        if generation_type == 'random_walk':
            SubG = findNeighborhoodFullRandom(G, start_node, WALK_LENGTH, NUM_WALKS, MAX_SIZE)
            center = findCenters(SubG)
            SubG,bfs_order,dependency = ordering_BFS(SubG,center)

        elif generation_type == 'BFS':
            SubG,bfs_order,dependency = neighborhood_BFS(G,start_node,BFS_DEPTH,MAX_SIZE)
        else:
            print('Invalid generation type')
                
        num_triangles = sum(nx.triangles(SubG).values()) / 3
        anom_triangles.append(num_triangles)
 
        #Generates the features of the graph
        adjacency_tensor,X = generateFeatures(SubG,bfs_order,node_feature_dict,EDGE_FEATURES,NODE_FEATURES)

        num_nodes = X.shape[0]
        anom_sizes[i] = num_nodes
        anom_matrices[i,:,:num_nodes,:num_nodes] = adjacency_tensor
        anom_features[i,:num_nodes] = X
        anom_degrees.append(float(2 * SubG.number_of_edges()) / num_nodes)




       
    total_degrees = np.array(total_degrees)
    total_triangles = np.array(total_triangles)
    anom_degrees = np.array(anom_degrees)
    anom_triangles = np.array(anom_triangles)
    
    #Discards all training samples that are too small
    
    train_matrices = train_matrices[train_sizes > 3]
    train_features = train_features[train_sizes > 3]
    total_degrees = total_degrees[train_sizes > 3]
    total_triangles = total_triangles[train_sizes > 3]
    train_sizes = train_sizes[train_sizes > 3]

    test_matrices = test_matrices[test_sizes > 3]
    test_features = test_features[test_sizes > 3]
    test_sizes = test_sizes[test_sizes > 3]

    anom_matrices = anom_matrices[anom_sizes > 3]
    anom_features = anom_features[anom_sizes > 3]
    extra_degrees = anom_degrees[anom_sizes > 3]
    anom_triangles = anom_triangles[anom_sizes > 3]
    anom_sizes = anom_sizes[anom_sizes > 3]
    

    print('Max dependency:',max_dependency)
    print('Training dataset node stats: mean - {:.2f} std - {:.2f}, [{:.2f},{:.2f}]'.format(np.average(train_sizes), np.std(train_sizes), np.amin(train_sizes),np.amax(train_sizes)))

    print('Training dataset degree stats: mean - {:.2f} std - {:.2f}, [{:.2f},{:.2f}]'.format(np.average(total_degrees), np.std(total_degrees), np.amin(total_degrees),np.amax(total_degrees)))

    print('Training dataset triangle stats: mean - {:.2f} std - {:.2f}, [{:.2f},{:.2f}]'.format(np.average(total_triangles), np.std(total_triangles), np.amin(total_triangles),np.amax(total_triangles)))


    os.chdir(os.getcwd() + '/test_data')
    np.save('train_adj_features.npy',train_matrices)
    np.save('train_node_features.npy',train_features)
    np.save('train_sizes.npy',train_sizes)
    print('Successfully generated',train_sizes.shape[0],'graphs for training')

    np.save('valid_adj_features.npy',valid_matrices)
    np.save('valid_node_features.npy',valid_features)
    np.save('valid_sizes.npy',valid_sizes)
    print('Successfully generated',valid_sizes.shape[0],'graphs for validation')

    np.save('test_adj_features.npy',test_matrices)
    np.save('test_node_features.npy',test_features)
    np.save('test_sizes.npy',test_sizes)
    #np.save('test_ids.npy',test_ids)
    #np.save('test_classes.npy',test_classes)
    print('Successfully generated',test_sizes.shape[0],'graphs for testing')


    print()
    np.save('extra_adj_features.npy',anom_matrices)
    np.save('extra_node_features.npy',anom_features)
    np.save('extra_sizes.npy',anom_sizes)

    print('Anomaly set node stats: mean - {:.2f} std - {:.2f}'.format(np.average(anom_sizes),np.std(anom_sizes)))
    print('Anomaly set degree stats: mean - {:.2f} std - {:.2f}'.format(np.average(anom_degrees),
    np.std(anom_degrees)))
    print('Anomaly set triangle stats: mean - {:.2f} std - {:.2f}'.format(np.average(anom_triangles),np.std(anom_triangles)))
    print('Successfully generated',anom_sizes.shape[0],'Anomalous graphs')
        

    configs = "{'max_size':" +  str(MAX_SIZE) + ",'node_dim':" + str(NODE_FEATURES)  + ",'edge_dim':" + str(EDGE_FEATURES) + "}"
    f = open('_config.txt', 'w')
    f.write(configs)
    f.close()