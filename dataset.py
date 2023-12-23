import torch
import math
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy import sparse


def get_dataset(num, args, device='cpu', path="./example_data/"):
    print('Loading {} dataset...'.format(args.dataset))
    path = path + args.dataset
    
    multiple_graph = []   
    for i in range(args.view_num):
        feat = sio.loadmat(path + "/feat" +str(i))['feat']
        feat = F_normalize_features(feat)
        adj = sio.loadmat(path + "/adj" +str(i))['W']
        if i==0:
            feature = feat
        else:
            feature = np.concatenate((feature, feat), axis=1)
        features = F_normalize_features(feature)
       
        adj = torch.FloatTensor(adj.todense()).to(device)
        multiple_graph.append(adj)
    
    features = torch.FloatTensor(np.array(features))
    labels = sio.loadmat(path + "/label")['label']
    labels = torch.LongTensor(labels.flatten())
    idx_train, idx_val, idx_test = train_valid_test_idx(path, labels.shape[0], num, args.train_ratio, args.valid_ratio)

    return multiple_graph, features.to(device), labels.to(device), idx_train.to(device), idx_val.to(device), idx_test.to(device)


def F_normalize_features(mx):
    """normalize feature matrix"""
    f_n = np.linalg.norm(mx,axis=1,keepdims=True)+1e-8
    mx = mx/f_n
    return mx


def train_valid_test_idx(path, node_num, num, train_ratio=.1, valid_ratio=.1):
    """ randomly splits label into train/valid/test splits """
    idx = np.loadtxt(path + '/' + str(int(train_ratio*100)) +'%/idx' + str(num), dtype='int')
    valid_num = math.ceil(node_num*valid_ratio)
    train_num = math.ceil(node_num*(1-valid_ratio)*train_ratio)
    
    train_idx = idx[:train_num] 
    valid_idx = idx[train_num:train_num + valid_num]  
    test_idx = idx[train_num + valid_num:]

    idx_train = torch.LongTensor(train_idx)
    idx_val = torch.LongTensor(valid_idx)
    idx_test = torch.LongTensor(test_idx)   

    return idx_train, idx_val, idx_test