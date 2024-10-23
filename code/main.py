import torch
import torch.nn as nn
from torch import optim
from model import LightGCN
from config import config
from dataloader import Loader
from time import time
import numpy as np

from utils import BPRLoss, BPRTrain


device = config['device']
dataset = Loader()
model = LightGCN(config, dataset).to(device)
bpr = BPRLoss(model, config)

epochs = 1
weight_file = 'weight_file'
print('start training:')
# print(UniformSample(dataset).shape)
# print(UniformSample(dataset)[0])
for epoch in range(epochs):
    start = time()
    if (epoch % 10 == 0) and (epoch != 0):
        print("[TEST]")
        # Test(dataset, model, epoch, config['multicore'])
    output_information = BPRTrain(dataset, model, bpr)
    print(f'EPOCH[{epoch+1}/{epochs}] {output_information}')
    torch.save(model.state_dict(), weight_file)

# output the chosen index
# alphas = model.user_embedding.alphas
# print(torch.argmax(alphas, dim=0))

# if __name__ == '__main__':
#     pass
# print(config['device'])
