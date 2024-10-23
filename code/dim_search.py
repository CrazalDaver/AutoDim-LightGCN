import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


# change embedding to fixed same dim
class DimTransform(nn.Module):
    def __init__(self, indim, outdim, trans_type='linear'):
        super().__init__()
        if trans_type == 'linear':
            self.transform = nn.Linear(indim, outdim)
        elif trans_type == 'zero padding':
            self.transform = nn.ZeroPad1d((0, outdim - indim))

    def forward(self, embed):
        return self.transform(embed)


class AutoDim_branch(nn.Module):
    def __init__(self, num_embeddings, dim, outdim=config['latent_dim']):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=dim)
        self.user_alpha = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=1)
        self.transform = DimTransform(indim=dim, outdim=outdim, trans_type='linear')
        self.bn = nn.BatchNorm1d(num_features=outdim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transform(x)
        x = self.bn(x)
        spec_alpha = self.user_alpha(x)  # alpha for each user
        return x, spec_alpha


class AutoDimEmbedding(nn.Module):
    """
    AutoDimEmbedding can replace the original embedding layers to try different embedding dims:
    candidate_dims
    num_embeddings = user_num from dataloader
    """

    def __init__(self, num_embeddings, embedding_dim=config['latent_dim']):
        super().__init__()
        self.candidate_dims = [16, 32, 64]  # v branch_num
        self.candidate_dims_num = len(self.candidate_dims)
        # parallel embeddings of different dims
        self.branches = nn.ModuleList([
            AutoDim_branch(num_embeddings=num_embeddings, dim=dim, outdim=embedding_dim) for dim in self.candidate_dims
        ])
        # self.alphas = nn.Parameter(torch.rand(
        #     self.candidate_dims_num,
        #     # config['batch_size'],
        #     # num_embeddings,  # input dimension: x_M
        # ))

    def getEmbedding(self):
        branch_output = []
        branch_alpha = []
        for i, br in enumerate(self.branches):  # iterate each branches
            emb_weight = br.embedding.weight
            user_alpha = br.user_alpha.weight
            branch_output.append(br.bn(br.transform(emb_weight)))
            branch_alpha.append(user_alpha)
        all_emb = torch.stack(branch_output) * F.softmax(torch.stack(branch_alpha), dim=0)
        return all_emb.sum(dim=0)  # # shape:(emb_num, emb_dim)

    def forward(self, x):   # x shape: [batch_size]
        # call getEmbedding and look up embeddings
        all_emb = self.getEmbedding()
        output = torch.stack([all_emb[i] for i in x])
        return output

# input_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int)
# print(AutoDimEmbedding(num_embeddings=5)(input_tensor).shape)
