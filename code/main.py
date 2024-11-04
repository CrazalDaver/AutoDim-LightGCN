import torch
import torch.nn as nn
from torch import optim
from model import LightGCN
from config import config
from dataloader import Loader
from time import time
import numpy as np

from utils import BPRLoss, BPRTrain, NegLogLikelihoodLoss, Test

# configuration
device = config['device']
dataset = Loader(dataset_name='amazon-book')
model = LightGCN(config, dataset).to(device)
bpr = BPRLoss(model, config)
neglog = NegLogLikelihoodLoss(model, config)
epochs = 1
weight_file = 'weight_file'

# training
print('start training:')
for epoch in range(epochs):
    start = time()
    print("[TEST]")
    Test(dataset, model, neglog)
    # if (epoch % 10 == 0) and (epoch != 0):
    #     print("[TEST]")
    #     Test(dataset, model, neglog)
    output_information = BPRTrain(dataset, model, bpr)
    print(f'EPOCH[{epoch+1}/{epochs}] {output_information}')
    torch.save(model.state_dict(), weight_file)

alpha_res = torch.stack([model.user_embedding.branches[i].user_alpha.weight
                         for i in range(model.user_embedding.candidate_dims_num)])
print(alpha_res.argmax(dim=0).squeeze())

# if __name__ == '__main__':
#     pass
# print(config['device'])

# retraining

