import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
import numpy as np

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

    def __init__(self, num_embeddings, embedding_dim=config['latent_dim'], learned_dims = None):
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
        self.learned_dims = learned_dims  # in re-training stage, learned_dims = list of learned_dims IDs else None
        self.num_embeddings = num_embeddings
    def getEmbedding(self):
        branch_output = []
        branch_alpha = []
        for i, br in enumerate(self.branches):  # iterate each branches
            emb_weight = br.embedding.weight
            user_alpha = br.user_alpha.weight
            branch_output.append(br.bn(br.transform(emb_weight)))
            branch_alpha.append(user_alpha)
        all_emb = torch.stack(branch_output) * F.softmax(torch.stack(branch_alpha), dim=0)
        return all_emb.sum(dim=0)  # shape:(emb_num, emb_dim)

    def finalEmbedding(self):  # for re-training stage
        branch_output = []
        for i, br in enumerate(self.branches):  # iterate each branches
            emb_weight = br.embedding.weight
            branch_output.append(br.bn(br.transform(emb_weight)))  # list of tensor(branch_output) for all cand_dims

        all_emb = []

        for i in range(self.num_embeddings):
            selected_branch = self.learned_dims[i]  # branch ID 0/1/2
            selected_embed = branch_output[selected_branch][i]
            all_emb.append(selected_embed)
        return torch.stack(all_emb)  # shape:(emb_num, emb_dim)

    def forward(self, x):   # x shape: [batch_size]
        if self.learned_dims is None:  # 1st stage;
            # call getEmbedding and look up embeddings
            all_emb = self.getEmbedding()
            output = torch.stack([all_emb[i] for i in x])
        else:
            all_emb = self.finalEmbedding()
            output = torch.stack([all_emb[i] for i in x])
        return output


if __name__ == '__main__':
    input_tensor = torch.tensor([i for i in range(10)])
    emb = AutoDimEmbedding(num_embeddings=input_tensor.shape[0], learned_dims=np.random.randint(0, 3, size=10))
    print(emb(input_tensor).shape)
    print(emb.finalEmbedding().shape)
