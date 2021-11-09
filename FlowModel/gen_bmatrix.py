#if needed
#!pip install ray 

import ray

""" Sample Usage:

pos_data = pd.read_csv("teleological-links-pos.csv", sep=",", header = None)

Here we use the first column and the third column

bmatrix, word_id, id_word, pos_id, id_pos = create_bmatrix_simple(pos_data[0], pos_data[2])

bmatrix is in the shape of N x 4 

N is the number of distinct words 

To have more features 

bmatrix, word_id, id_word, pos_id, id_pos = create_bmatrix(pos_data[0], pos_data[2])

bmatrix is now N x 9; the combination of pos is taken into account

The ray package is used for parallelization of loading the dictionaries; 
it is not necessary since data set is small

"""

@ray.remote
def load_word_dict(target):
    index = 0
        
    word_id_dict = {}
    id_word_dict = {}
        
    for entry in target.unique():
        if (entry not in word_id_dict):
            word_id_dict[entry] = index
            id_word_dict[index] = entry
            index += 1
        
    return word_id_dict, id_word_dict

@ray.remote
def load_pos_dict(target):
    index = 0

    pos_id_dict = {}
    id_pos_dict = {}

    for entry in target.unique():
        if (entry not in pos_id_dict):
            pos_id_dict[entry] = index
            
            id_pos_dict[index] = entry
            
            index += 1

    return pos_id_dict, id_word_dict

def create_bmatrix(col1, col2):
    assert(col1.shape[0] == col2.shape[0])

    rows = len(col1.unique())
    cols = len(col2.unique())
    
    bin_matrix = np.zeros((rows, cols))

    """ Load word and pos dictionaries concurrently """
    f1 = load_pos_dict.remote(col2)
    f2 = load_word_dict.remote(col1)
    
    ret1, ret2 = ray.get([f1, f2])
    
    """pos_id contains a part of speech and its id, e.g., 'n' -> 0 
    
       id_pos contains an id and its part of speech
       
       the same applies to word_id and id_word
    """
    
    pos_id, id_pos = ret1
    
    word_id, id_word = ret2
    
    """ Set entry in the matrix equal to 1.0 if matches """
    for j in range(col1.shape[0]):
        bin_matrix[ word_id[ col1[j] ] ] [ pos_id[ col2[j] ] ] = 1.0
    
    return bin_matrix, word_id, id_word, pos_id, id_pos

def create_bmatrix_simple(col1, col2):
    assert(col1.shape[0] == col2.shape[0])
    """Return a simple binary matrix here"""
    
    rows = len(col1.unique())
    
    pos_id = {'n' : 0, 'v' : 1, 'a' : 2, 'r' : 3 }
    id_pos = { 0 : 'n', 1 : 'v', 2 : 'a', 3 : 'r'}
    
    bin_matrix = np.zeros((rows, 4))
    
    f = load_word_dict.remote(col1)
    
    ret = ray.get(f)
    
    word_id, id_word = ret
    
    for j in range(col1.shape[0]):
        pos = col2[j].split(',')
        
        for p in pos:
            bin_matrix[ word_id[ col1[j] ] ] [ pos_id[ p ] ] = 1.0
    
    return bin_matrix, word_id, id_word, pos_id, id_pos
