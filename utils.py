import torch
import torch.nn as nn
from torch import optim
from model import LightGCN
from config import config
from dataloader import Loader
from time import time
import numpy as np


class BPRLoss:
    def __init__(self,
                 model: LightGCN,
                 config: config):
        self.model = model
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(model.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.getLoss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


def UniformSample(dataset):
    """
    the original implement of BPR Sampling in LightGCN
    :return: np.array
    """
    # start = time()
    user_num = dataset.trainDataSize  # 109930 interactions
    users = np.random.randint(0, dataset.n_user, size=user_num)  # (low, high=None, size=None, dtype=int)
    allPos = dataset.allPos  # user-positive items for all users
    S = []

    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:  # in case=0
            continue
        posindex = np.random.randint(0, len(posForUser))  # 1 random positive item
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_item)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
    return np.array(S)


def shuffle(*arrays, **kwargs):
    # default
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def minibatch(*tensors, **kwargs):  # batch iterator
    # default
    batch_size = kwargs.get('batch_size', config['batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            if i + batch_size <= len(tensor):
                yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            if i + batch_size <= len(tensors[0]):
                yield tuple(x[i:i + batch_size] for x in tensors)


# freeze model & unfreeze alpha
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    model.user_embedding.alphas.requires_grad = True


# freeze alpha & unfreeze model
def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    model.user_embedding.alphas.requires_grad = False


def BPRTrain(dataset, model, loss_type, epoch=1, neg_k=1):
    # model = model
    model.train()
    # unfreeze(model)
    bpr: BPRLoss = loss_type

    start_time = time()

    S = UniformSample(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    device = config['device']

    users = users.to(device)
    posItems = posItems.to(device)
    negItems = negItems.to(device)
    users, posItems, negItems = shuffle(users, posItems, negItems)
    total_batch = len(users) // config['batch_size'] + 1

    average_loss = 0.
    for (batch_i,
         (batch_users, batch_pos, batch_neg)) in enumerate(minibatch(users, posItems, negItems,
                                                                     batch_size=config['batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        average_loss += cri

    average_loss = average_loss / total_batch
    epoch_time = time() - start_time
    return f"Training loss: {average_loss:.3f} - epoch_time: {epoch_time}"


# update the alpha and evaluate the model
def Test(dataset, model):
    batch_size = config['test_batch_size']


# update alpha values
    model.train()
    freeze(model)




