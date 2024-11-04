import torch
import torch.nn as nn
from torch import optim
from model import LightGCN
from config import config
from dataloader import Loader
from time import time
import numpy as np
from sklearn.metrics import roc_auc_score


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


class NegLogLikelihoodLoss:
    """
    negative log-likelihood function
    """

    def __init__(self,
                 model: LightGCN,
                 config: config):
        self.model = model
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(model.parameters(), lr=self.lr)

    def stageOne(self, y_pred, y_truth):
        criterion = nn.BCELoss()
        # y_truth = y_truth.detach()
        loss = criterion(y_pred, y_truth)
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
        # end = time()
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
    # model.user_embedding.alphas.requires_grad = True
    for br in model.user_embedding.branches:
        br.user_alpha.requires_grad = True


# freeze alpha & unfreeze model
def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    # model.user_embedding.alphas.requires_grad = False
    for br in model.user_embedding.branches:
        br.user_alpha.requires_grad = False


def BPRTrain(dataset, model, loss_type, epoch=1, neg_k=1):
    # model = model
    model.train()
    unfreeze(model)
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


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def RecallPrecision(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precision = np.sum(right_pred) / precis_n
    # return recall, precision
    return {'recall': recall, 'precision': precision}


def NDCG(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in config['top_k']:
        ret = RecallPrecision(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCG(groundTrue,r,k))
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg)}


# update the alpha and evaluate the model
def Test(dataset, model, loss_type, config=config):
    batch_size = config['test_batch_size']
    model: model.LightGCN
    testDict: dict = dataset.testDict  # dict:{user:positive items}
    neglog: NegLogLikelihoodLoss = loss_type

    top_k = config['top_k']
    device = config['device']

    results = {'precision': np.zeros(len(top_k)),
               'recall': np.zeros(len(top_k)),
               'ndcg': np.zeros(len(top_k))}

    users = list(testDict.keys())

    # update alpha values
    model.train()
    freeze(model)
    users_list = []
    rating_list = []
    groundTrue_list = []
    losses = []

    for batch_users in minibatch(users, batch_size=batch_size):
        allPos = dataset.getUserPosItems(batch_users)  # (user: positive items) pairs for training set
        groundTruth = [testDict[u] for u in batch_users]  # (user: positive items) pairs for test set
        batch_users_gpu = (torch.Tensor(batch_users).long()).to(device)  # users in batch
        ratings = model.getUsersRating(batch_users_gpu)
        gt = torch.zeros(ratings.shape, requires_grad=True).to(device)
        for i, line in enumerate(gt):
            idxes = [idx for idx in groundTruth[i] if idx < dataset.m_item]
            line = line.clone()
            line[idxes] = 1
        loss = neglog.stageOne(ratings, gt)
        losses.append(loss)

        # exclude data in training set
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        ratings[exclude_index, exclude_items] = -1024
        # _, rating_K = torch.topk(ratings, k=max(top_k))
        rating_K = torch.topk(ratings, max(top_k)).indices  # shape: (user_num, top_k_max)
        ratings = ratings.cpu().numpy()

        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTruth)

    pre_results = []
    X = zip(rating_list, groundTrue_list)
    for x in X:
        pre_results.append(test_one_batch(x))
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    results['recall'] /= float(len(users))
    results['precision'] /= float(len(users))
    results['ndcg'] /= float(len(users))

    results['neglog_loss'] = np.mean(losses)

    return results
