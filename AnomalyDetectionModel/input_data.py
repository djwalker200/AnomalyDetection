import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np

import snap
import math
import random
import scipy
from random import sample

import os
import sys
import json

from time import time

# Converts the node feature matrix into a dictionary
#
# Parameters:
# feature_matrix - matrix containing node features in which each row corresponds to the features of a node
# indexed - boolean variable denoting whether or not the first column of the feature matrix contains node indices           
# class_matrix - matrix that stores the class of each node
# name_converter - dictionary that converts integer class tags to string labels (optional) 
# Output:
# feature_dict - dictionary of form (node ID : <attribute vector>)
# class_dict - dictionary of form (node ID : class label)
def setupFeatures(feature_matrix,indexed=True,class_matrix=None,name_converter = None):

    feature_dict = {}
    class_dict = {}
    mtx = feature_matrix[:,1:-1]
   
    if indexed:

        for i in range(feature_matrix.shape[0]):
            
            idx = feature_matrix[i,0]
            #Stores the pair of (node ID : <attribute vector>)
            feature_dict[idx] = mtx[i,:]
            #Stores the pair of (node ID : class label)
            cl = feature_matrix[i,-1]
            if name_converter is not None:
                class_dict[idx] = name_converter[cl]
            else:
                class_dict[idx] = cl
    else:
        for i in range(feature_matrix.shape[0]):
            feature_dict[i] = feature_matrix[i].astype(np.float32)
            cl = class_matrix[i]
            if name_converter is not None:
                class_dict[idx] = name_converter[cl]
            else:
                class_dict[i] = cl


    return feature_dict,class_dict

# Returms the class labels of an ordered set of nodes
#
# Parameters:
# node_order - An ordered list of nodes referenced by their ID
# label_dict - A dictionary with node IDs as keys and class labels as values
# Output:
# label_array - A numpy array containing the ordered class labels of the nodes in node_order
def getLabels(node_order,label_dict):
        
    labels = []
    for j,node in enumerate(node_order):
        #Finds the class label for the given node, and asserts the node has a class
        lb = label_dict.get(node)
        if lb is not None:
            labels.append(lb)
        else:
            labels.append('0')

    #Converts to a numpy string array
    return np.array(labels)

# Finds the neighborhood of a node u in a graph G by implementing breadth first search up to N layers
# 
# Parameters:
# G - An undirected networkx graph
# u - The starting node in the graph for BFS
# depth - The BFS tree depth at which the search will terminate
# Output:
# SubG - A BFS networkx subgraph of G starting from node u
# visited - An ordered list containing all nodes visited during BFS
# max_dependency - Maximum BFS layer dependency
def neighborhood_BFS(G,u,depth,MAX_SIZE):

    #Counts the number of visited nodes
    node_counter = 0
    #BFS parameters
    visited = []
    queue = [(u,0)]
    
    #Variables for tracking max dependency
    farthest_dist = 0
    last_dist_idx = 0
    dependency = 0

    while(queue):

        #Takes next node in BFS ordering from FIFO queue
        s,distance = queue.pop(0)

        #Checks for depth limit
        if distance > depth:
            break

        #Stores the node as visited if not already visited
        if s in visited:
            continue

        node_counter += 1
        visited.append(s)

        #Updates the BFS dependency of the graph
        if(distance > farthest_dist):
            dependency = max(dependency, node_counter - last_dist_idx)
            farthest_dist = distance
            last_dist_idx = node_counter

        #Checks for size constraint
        if(node_counter == MAX_SIZE):
            break

        #Gets the nodes neighbors and adds to queue
        neighborhood_dict = G[s]
        for neighbor in neighborhood_dict:
            if neighbor not in visited:
                queue.append((neighbor,distance + 1))

    
    dependency = max(dependency, node_counter - last_dist_idx)


    #Creates a subgraph of the visited nodes
    SG = G.subgraph(visited)
    
    #Handles reodreding
    SubG = nx.OrderedGraph()
    SubG.add_nodes_from(visited)
    SubG.add_edges_from((u, v) for (u, v) in SG.edges() if u in SubG if v in SubG)

    #Returns the neighborhood subgraph and the bfs node ordering
    return SubG,visited,dependency

# Finds a BFS ordering for a graph G with a starting node u 
# Initial conditions should be met that |V(G)| < Max size and u should be a central node in G
#
#Parameters:
# G - undirected networkx graph
# u - starting node for BFS
#Output:
# G_prime - The original graph G with vertices reordered
# visited - An ordered list containing all nodes visited during BFS
def ordering_BFS(G,u):

    #List of all nodes traversed
    visited = []
    #BFS Queue
    queue = [(u,0)]

    #Variables for tracking max dependency
    farthest_dist = 0
    last_dist_idx = 0
    dependency = 0
    node_counter = 0
    '''
    #Boolean dictionary indicating whether each node was seen in BFS traversal
    # (Should only be false for disconnected nodes)
    found = {}
    for node in G:
        found[node] = False
    '''

    #Runs BFS
    while(queue):
        s,distance = queue.pop(0)

        #Updates the BFS dependency of the graph
        if(distance > farthest_dist):
            dependency = max(dependency, node_counter - last_dist_idx)
            farthest_dist = distance
            last_dist_idx = node_counter

        if s in visited:
            continue

        visited.append(s)
        node_counter += 1

        #found[s] = True
        #Gathers neighbors of s and adds to queue if not yet seen
        neighborhood_dict = G[s]
        for neighbor in neighborhood_dict:
            if neighbor not in visited:
                queue.append((neighbor,distance + 1))
                
    dependency = max(dependency, node_counter - last_dist_idx)

    #Handles reodreding
    SubG = nx.OrderedGraph()
    SubG.add_nodes_from(visited)
    SubG.add_edges_from((u, v) for (u, v) in G.edges() if u in SubG if v in SubG)

    '''
    for node in G:
        if not found[node]:
            G_prime.add_node(node)
            visited.append(node)
    '''

    #Returns the neighborhood subgraph and the bfs node ordering
    return SubG,visited,dependency


# Finds the node in a graph G that has the highest eigenvalue centrality value
#
# Parameters:
# G - An undirected networkx graph
# draw - True if the graph should be drawn with the center colored 
# Output:
# center - the index of the center node
def findCenters(G,draw=False):
    '''
    #Calculates eigenvector centrality measure for each node
    eigen_centrality = nx.eigenvector_centrality_numpy(G)

    #Calculates eigenvector center
    eigen_keys = list(eigen_centrality.keys())
    eigen_values = np.array(list(eigen_centrality.values()))
    idx = np.argmax(eigen_values)
    eigen_center = eigen_keys[idx]
    '''

    #Calculates closeness centrality measure for each node
    closeness_centrality = nx.closeness_centrality(G)

    #Calculates closeness center
    closeness_keys = list(closeness_centrality.keys())
    closeness_values = np.array(list(closeness_centrality.values()))
    idx = np.argmax(closeness_values)
    closeness_center = closeness_keys[idx]

    #Draws the centers if draw parameter is specified
    if draw:

        color_map = []
        for node in G:
            if node == eigen_center and node == closeness_center:
                color_map.append('purple')
            elif node == eigen_center:
                color_map.append('red')
            elif node == closeness_center:
                color_map.append('blue')
            else:
                color_map.append('green')

        nx.draw(G,node_color=color_map,with_labels=True)
        #plt.savefig('center.png')
        plt.show()
        plt.clf()

    return closeness_center

# Finds the neighborhood of a node u by taking the union of a series of random walks from u
# The union is taken on ONLY the nodes of the random walks
#
# Parameters:
# G - Input Graph
# u - The starting node to generate the neighborhood of
# walk_length - The length of the random walk
# num_walks - The number of random walks 
# Output:
# SubG - The generated neighborhood subgraph on G from the start node u

def findNeighborhoodFullRandom(G,u,walk_length,num_walks,max_nodes):
    #List to store the set of nodes in the neighborhood of u
    #Originally stores just the start vertex u
    neighborhood = [u]
    #Counts the number of nodes in the neighborhood
    node_counter = 1
    MAX_REACHED = False
    #Loop that controls the number of walks
    for iteration in range(num_walks):
        #Asserts that all walks start from the node u
        current_node = u
        #Asserts that the graph is less than the maximum size allowed
        if MAX_REACHED:
            break
        #Loops through each step of the walk
        for step in range(walk_length):
            #Stores the adjacency dictionary for the current node in the walk
            neighborhood_dict = G[current_node]
            #Stores the degree of the current node in the walk
            deg = len(neighborhood_dict)
            #Randomly generates a number in the range (0,1)
            p = random.random()
            #Uniformly selects an integer in the range [0,deg - 1] that represents the 
            #Next node in the random walk
            rand_index = int(p * deg)

            #Generates the node index of the next node in the walk by taking the rand_index'th 
            #node adjacent to the current node in the walk
            next_node = list(neighborhood_dict)[rand_index]

            #Adds the next node to the neighborhood node set if not already present
            if next_node not in neighborhood:
                neighborhood.append(next_node)
                node_counter += 1

            #Stops process if max graph size has been reached
            if node_counter == max_nodes:
                MAX_REACHED = True
                break
            #Updates the current node in the walk
            current_node = next_node

    #Generates the subgraph of G containing all nodes in the detected neighborhood
    SubG = G.subgraph(neighborhood)
    #Returns the neighborhood subgraph
    return SubG

# Converts an adjacency matrix to the corresponding adjacency tensor and node feature matrix
#
# Parameters:
# A - Adjacency matrix of a graph
# num_nodes - Number of nodes in the graph of A
# EDGE_FEATURES - The number of desired edge features
# NODE_FEATURES - The number of desired node features
# Output:
# adjacency_tensor - Adjacency tensor for the given graph (EDGE_FEATURES x num_nodes x num_nodes)
# X - Node feature matrix for the given graph (num_nodes x NODE_FEATURES)
def generateFeatures(A,node_order,feature_dict,EDGE_FEATURES,NODE_FEATURES):

    num_nodes = len(node_order)


    #SHOULD BE USED
    if EDGE_FEATURES == 1:

        #A = np.expand_dims(A,axis=0)
        #adjacency_tensor = np.concatenate((A,1-A),axis=0)
        adjacency_tensor = A
    else:
        print('Edge Features != 1 Not supported')
        exit()
  
    #Used if nodes are attributed
    if NODE_FEATURES > 1:
        
        X = np.zeros((num_nodes,NODE_FEATURES))
        for j,node in enumerate(node_order):
            feature = feature_dict.get(node)
            if feature is not None:
                X[j,:] = feature    

    #Used if nodes are unattributed
    elif NODE_FEATURES == 1:
        X = np.ones(num_nodes)
        X = np.expand_dims(X,1)
    else:
        print('No node features')
        exit()

    #Returns the adjacency matrix and node feature matrix
    return adjacency_tensor,X

# Adds random cliques to the structure of a graph
#
# Parameters:
# G - Input networkx graph that will be added to
# num_cliques - number of cliques to generate
# clique_size - number of nodes for each clique
# Output:
# G - modified graph with cliques added
# nodes - list of all of the nodes in the generated cliques
def AddCliques(G,num_cliques,clique_size):
    #Creates a random set of nodes in G
    nodes = sample(list(G.nodes()), num_cliques * clique_size)

    for n in range(num_cliques):
        #Takes the next m nodes in the ordering
        clique_nodes = nodes[n * clique_size : (n + 1) * clique_size]
        #Forms a clique between all m nodes
        for i in range(clique_size):
            for j in range(i,clique_size):
                G.add_edge(clique_nodes[i],clique_nodes[j])
    #Returns the modified graph and all nodes that were modified
    return G,nodes

# Scrambles the attributes of N nodes in G
#
# Parameters:
# G - Graph for which the features should be scrambled
# feature_dict - node feature dictionary for the given graph (node ID, <features>)
# N - Number of nodes to scramble
# NODE_FEATURES - Number of node attributes
# Output:
# feature_dict - Updated node feature dictionary
# nodes - List of nodes that had their features randomized
def Scramble(G, feature_dict, N, NODE_FEATURES,p=0.5):
    nodes = sample(list(G.nodes()), N)
    for node_id in nodes:
        rand_features = np.random.rand(1,NODE_FEATURES)
        rand_features[rand_features <= p] = 0
        rand_features[rand_features > p] = 1
        feature_dict[node_id] = rand_features

    return feature_dict,nodes

# Creates anomalies by sampling equally sized neighborhoods from two networks
# and swapping the structure of the two neighborhoods
#
# Parameters:
# G - Original graph that was used for training
# H - New graph to sample structural patterns from to derive anomalies
# N - Number of anomalies to generate
# MAX_SIZE - maximum neighborhood size
# NODE_FEATURES - Number of node features
# feature_dict - dictionary for node features <node, feature vector>
# Output
# Adjacency matrices,feature matrices, neighborhood sizes, average degrees and average triangles

def GenerateSwapAnomalies(G,H,N,MAX_SIZE,NODE_FEATURES,feature_dict):
    
    random_nodes = sample(list(G.nodes()),N)
    H_random_nodes = sample(list(H.nodes()),N)
    
    anomaly_sizes = np.zeros(N)
    anomaly_features = np.zeros((N,MAX_SIZE,NODE_FEATURES))
    anomaly_adj = np.zeros((N,MAX_SIZE,MAX_SIZE))
    degrees = []
    triangles = []
    for i,start_node in enumerate(random_nodes):
        H_start = H_random_nodes[i]

        SubG,bfs_order,__ = neighborhood_BFS(G,start_node,2,MAX_SIZE)


        N = len(bfs_order)
        SubH,H_bfs,__ = neighborhood_BFS(H,H_start,3,N)

        n = len(H_bfs)
        bfs_order = bfs_order[:n]

        num_triangles = sum(nx.triangles(SubH).values()) / 3
        triangles.append(num_triangles)
        degrees.append((float(2 * SubH.number_of_edges()) / n))

        A = nx.to_numpy_matrix(SubH)

        adj,X = generateFeatures(A,bfs_order,feature_dict,1,NODE_FEATURES)

        anomaly_adj[i,:n,:n] = adj
        anomaly_features[i,:n] = X
        anomaly_sizes[i] = n

    return anomaly_adj,anomaly_features,anomaly_sizes,degrees,triangles

# Takes a list of nodes and derives an anomolous feature vector for them that is
# sampled from a node of another class
#
# Parameters:
# G - networkx Graph to sample from (training graph)
# anomaly_nodes - list of anomaly nodes to swap features for
# NODE_FEATURES - Number of node attributes
# class_dict - dictionary of form <node, class ID>
# feature_dict - dictionary of form <node, feature vector>
# Output:
# new_features - (N x NODE_FEATURES) matrix where each row stores the new feature vector
#for the original node
def FeatureSwapAnomalies(G,anomaly_nodes,NODE_FEATURES,class_dict,feature_dict):
    
    new_features = np.ones((len(anomaly_nodes),NODE_FEATURES))
    for i,node in enumerate(anomaly_nodes):
        new_features[i,int(NODE_FEATURES / 4) :] = 0
        continue
        cl = class_dict.get(node)
        if cl is None:
            continue
        else:
            random_nodes = sample(list(G.nodes()),10)
            for r in random_nodes:
                rcl = class_dict.get(r)
                if rcl is None or rcl == cl:
                    continue
                else:
                    new_features[i] = feature_dict[r]
                    break

                
    return new_features
