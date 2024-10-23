"""for dataset like amazon-book and gowalla
set u-i interactions in structure of:

(uid items-he-react-with)
0 1 2 3 ...
1 2 3 84...
2 10 100...
"""

import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from time import time
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from config import config

# from torch.sparse import FloatTensor
# path = "F:/desktop/LightGCN-PyTorch-master/test implementation/data/"
current_directory = os.getcwd().replace('\\', '/')
path = os.path.join(current_directory, 'data/').replace('\\', '/')


def read_file(file_path, max_len=1000):
    unique_users_list, users_list, items_list = [], [], []
    with open(file_path) as f:
        for i, line in enumerate(f.readlines()):
            if (len(line) > 0) and (i < max_len):
                l = line.strip('\n').strip(' ').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                unique_users_list.append(uid)
                users_list.extend([uid] * len(items))
                items_list.extend(items)
    #                 m_item = max(m_item, max(items))
    #                 n_user = max(n_user, uid)
    #                 trainDataSize += len(items)

    return unique_users_list, items_list, users_list


class Loader(Dataset):
    def __init__(self, config=config, path=path, dataset_name='amazon-book'):
        train_file = path + dataset_name + '/train.txt'
        test_file = path + dataset_name + '/test.txt'
        # self.split = config['A_split']
        self.folds = config['n_fold']
        self.device = config['device']
        # self.mode_dict = {'train': 0, "test": 1}
        # self.mode = self.mode_dict['train']
        self.n_user = 0  # user count
        self.m_item = 0  # item count
        self.path = path
        # trainUniqueUsers, trainItem, trainUser = [], [], []
        # testUniqueUsers, testItem, testUser = [], [], []
        self.trainDataSize = 0
        self.testDataSize = 0
        self.Graph = None
    # I have write the file reading part into function because it is cool
        # # read training data file
        # with open(train_file) as f:
        #     for line in f.readlines():
        #         if len(line) > 0:
        #             l = line.strip('\n').strip(' ').split(' ')
        #             items = [int(i) for i in l[1:]]
        #             uid = int(l[0])
        #             trainUniqueUsers.append(uid)
        #             trainUser.extend([uid] * len(items))
        #             trainItem.extend(items)
        #             self.m_item = max(self.m_item, max(items))
        #             self.n_user = max(self.n_user, uid)
        #             self.trainDataSize += len(items)
        # # transform into
        # self.trainUniqueUsers = np.array(trainUniqueUsers)
        # self.trainUser = np.array(trainUser)
        # self.trainItem = np.array(trainItem)
        # # read test data file
        # with open(test_file) as f:
        #     for line in f.readlines():
        #         if len(line) > 0:
        #             l = line.strip('\n').strip(' ').split(' ')
        #             items = [int(i) for i in l[1:]]
        #             uid = int(l[0])
        #             testUniqueUsers.append(uid)
        #             testUser.extend([uid] * len(items))
        #             testItem.extend(items)
        #             self.m_item = max(self.m_item, max(items))
        #             self.n_user = max(self.n_user, uid)
        #             self.testDataSize += len(items)

        # read file
        # training dataset
        trainUniqueUsers, trainItem, trainUser = read_file(train_file)
        # self.n_user = max(trainUniqueUsers) + 1  # largest uid
        # self.m_item = max(trainItem) + 1  # id start from 0
        self.n_user = len(set(trainUniqueUsers))
        self.m_item = len(set(trainItem))
        self.trainDataSize = len(trainItem)  # len(trainItem) = len(trainUser), interaction count
        # into array
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        # test dataset
        testUniqueUsers, testItem, testUser = read_file(test_file)
        self.testDataSize = len(testItem)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

    # print some dataset info
        # print(f"{self.trainDataSize} interactions for training")
        # print(f"{self.testDataSize} interactions for testing")
        print(f"{dataset_name} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_user / self.m_item}")

        # (users,items), bipartite graph
        # csr_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols))
        # print('matrix shape', len(self.trainUser))
        # print('index length', (len(self.trainUser), len(self.trainItem)))
        # print('n', self.n_user)
        # print('m', self.m_item)
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # pre-calculate
        self.allPos = self.getUserPosItems(list(range(self.n_user)))
        # self.__testDict = self.__build_test()
        print(f"{dataset_name} initialization complete.")

    # fetch positive items
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems  # posItems[user] = positive items of user

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        # return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), dtype=torch.float32)

    # get adjacency matrix
    def getSparseGraph(self):
        """adjacency matrix
        adj_mat =
        |0(n)    R   |
        |R.T     0(m)|
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:  # try to load if calculated
                pre_adj_mat = sp.load_npz(self.path + 's_pre_adj_mat.npz')
                print("successfully loaded")
                norm_adj = pre_adj_mat
            except:
                print("calculating adjacency matrix")
                start = time()
                # create sparse adjacency matrix of shape (m+n,m+n)
                adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32)
                adj_mat = adj_mat.tolil()  # List of Lists
                R = self.UserItemNet.tolil()  # (u,i) bipartite graph
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()  # Dictionary of Keys
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                # Laplace Matrix & Normalization
                row_sum = np.array(adj_mat.sum(axis=1))  # sum: degree matrix
                d_inv = np.power(row_sum, -0.5).flatten()  # ^(0.5): degree inverse
                d_inv[np.isinf(d_inv)] = 0.  # in case degree=0, let inf=0
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - start}s, saved norm_mat...")
                sp.save_npz(self.path + 's_pre_adj_mat.npz', norm_adj)

            # no splitting
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)
            # print("don't split the matrix")
        return self.Graph

# sample_dataset = Loader()
# print(sample_dataset.device)
# print(sample_dataset.getSparseGraph())
# print(type(sample_dataset.Graph))
# print(type(sample_dataset.getSparseGraph()))
# print((sample_dataset().allPos)[0])
