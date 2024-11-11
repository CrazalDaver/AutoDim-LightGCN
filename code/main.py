import torch
import torch.nn as nn
from torch import optim
from model import LightGCN
from config import config
from dataloader import Loader
from time import time
import numpy as np

from utils import BPRLoss, BPRTrain, NegLogLikelihoodLoss, Test, dimSearchTrain

if __name__ == '__main__':
    # configuration
    device = config['device']
    dataset = Loader(dataset_name='amazon-book')
    model = LightGCN(config, dataset).to(device)
    bpr = BPRLoss(model, config)
    neglog = NegLogLikelihoodLoss(model, config)
    epochs = 1
    weight_file = 'weight_file'
    result = {}
    # training
    print('start auto-dim training:')
    neglog_losses = []
    bpr_losses = []
    for epoch in range(epochs):
        neglog_loss, bpr_loss = dimSearchTrain(dataset, model, bpr, neglog, config)
        neglog_losses.append(neglog_loss)
        bpr_losses.append(bpr_loss)
    print('auto-dim training finished:')

    # finish 1st dimension choice stage
    alpha_res = torch.stack([model.user_embedding.branches[i].user_alpha.weight
                             for i in range(model.user_embedding.candidate_dims_num)])
    # print(alpha_res.argmax(dim=0).squeeze())
    dim_result = alpha_res.argmax(dim=0).squeeze().cpu().numpy()  # result of dimensions the model chose
    # print(dim_result)

    # retraining
    model = LightGCN(config, dataset, learned_dims=dim_result).to(device)
    print('start re-training:')
    start = time()
    recalls = []
    precisions = []
    ndcgs = []
    for epoch in range(epochs):
        # test
        if (epoch % 10 == 0) and (epoch != 0):
            print("[TEST]")
            res = Test(dataset, model, neglog)
            recalls = res['recall']
            precisions = res['precision']
            ndcgs = res['ndcg']

        # train
        epoch_loss, epoch_time = BPRTrain(dataset, model, bpr)
        output_information = f"Training loss: {epoch_loss:.3f} - epoch_time: {epoch_time}"
        print(f'EPOCH[{epoch + 1}/{epochs}] {output_information}')
