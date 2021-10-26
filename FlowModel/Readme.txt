
AnomalyAF is our autoregressive normalizing flow model for anomaly detection. The base of the code is taken from the graphAF implementation. There are two steps in generating results with the code in this directory.

1) Convert a network into sampled neighborhoods to serve as input to the flow model
2) Run the flow model on the preprocessed data

These two steps can be accomplished as follows:

1) Two input files are required to generate the sampled neighborhoods for training our model. For a network with title NETWORK_NAME there should be a folder titled NETWORK_NAME in the same folder as the source code. Inside of this folder there should be a file containing the edge list of the network. This file should have exactly two entries on each line corresponding to two node IDs that share an edge. This file should be named NETWORK_NAME_edges.txt. An example of the formatting for this file is shown below:

node 1 ID:      node 2 ID:

0               1
0               2
0               3
1               2
.               .
.               .
.               .

The other file that is required is the node feature matrix. The first column of this matrix should denote the node IDs and the following D columns should represent the corresponding node attributes. Then optionally the last column should denote the node class if applicable. This file should be named NETWORK_NAME_fetures.txt. An example is displayed below:

node ID:        feature 1:        feature 2:        ...       feature D:        class label:
0               0                 1                 ...       1                 'ML'
1               1                 0                 ...       0                 'ML'
2               0                 0                 ...       0                 'AI'

Once these files are in the folder NETWORK_NAME in the AnomalyAF directory, the features can be generated with the following command line. The features will then be stored as .npy files in a folder named test_data. In the following command line, the extra arguments are necessary and determine the number of samples for the training, validation, and testing sets respectively.

    python3.8 data_process.py NETWORK_NAME <n_train> <n_valid> <n_test>
    
 2) After generating the sampled neighborhoods, the model can be run with the following standard command line. Note that there are additional command line options that can be found in the arguments section of train.py (lines 250 - 307). The argument n_epochs controls the number of training epochs, and if not provided has a default value of 20 epochs.
 
    python3.8 train.py --path test_data --train --epochs <n_epochs> --learn_prior
 
 
 This concludes the process for running our model, and this command line argument will summarize the model performance by outputting the loss value every 5 epochs, as well as the following statistics:
 
 AUC
 F1 Score
 Precision@50
 Recall@50
 Precision@100
 Recall@100
 Precision
 Recall
    
    
