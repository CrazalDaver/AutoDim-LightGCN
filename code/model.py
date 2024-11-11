import torch
from torch import nn as nn
from dataloader import Loader
# setting model parameters
from config import config
from dim_search import AutoDimEmbedding


class LightGCN(nn.Module):
    def __init__(self,
                 config: dict,
                 dataset: Loader,
                 use_AutoDim=True,
                 learned_dims = None
                 ):
        super(LightGCN, self).__init__()
        self.config = config  # parameters
        self.dataset = dataset
        self.use_AutoDim = use_AutoDim
        self.learned_dims = learned_dims
        self.__init_weight()  # load parameters

        # self.Embedding = AutoDimEmbedding if self.use_AutoDim else torch.nn.Embedding

    def __init_weight(self):
        self.num_users = self.dataset.n_user
        self.num_items = self.dataset.m_item
        self.latent_dim = self.config['latent_dim']
        self.n_layers = self.config['n_layers']
        self.graph = self.dataset.getSparseGraph()

        # self.dim_search = AutoDimEmbedding(
        #     num_embeddings=self.num_users, embedding_dim=self.latent_dim)  # AutoDim similar to nn.Embedding
        # self.user_embedding = torch.nn.Embedding(
        #     num_embeddings=self.num_users, embedding_dim=self.latent_dim)

        # embedding = AutoDimEmbedding if self.use_AutoDim else torch.nn.Embedding

        self.user_embedding = AutoDimEmbedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim, learned_dims=self.learned_dims)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

    def propagate(self):
        """propagate methods for lightGCN
        in matrix form, each layer all-embedding matrix is multiplied with degree matrix
        E^(k+1)=[D^(-1/2) A D^(-1/2)] E^(k)

        :return: user & item embeddings
        """
        # users_emb = self.user_embedding.weight
        users_emb = self.user_embedding.getEmbedding()
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])  # (num_embeddings=num_users+num_items, embedding_dim=latent_dim)
        embs = [all_emb]  # record embedding from each layer

        # graph dropout, maybe just not drop

        # iteration for propagation
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)  # new-dimension concatenating
        final_emb = torch.mean(embs, dim=1)  # final embed = weighted sum(mean) of all layers' embed
        users, items = torch.split(final_emb, [self.num_users, self.num_items])
        return users, items

    # predict all ratings of given user
    def getUsersRating(self, users):
        all_users, all_items = self.propagate()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = nn.Sigmoid()(torch.matmul(users_emb, items_emb.t()))
        return rating

    # fetch all embeddings for loss calculation
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.propagate()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.user_embedding(users)
        pos_emb_ego = self.item_embedding(pos_items)
        neg_emb_ego = self.item_embedding(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    # calculate loss: BPR loss and regulation term
    def getLoss(self, users, pos, neg):
        """ Bayesian Personalized Ranking(BPR) loss
        BPR loss= - sum(ln(activation(pred_pos - pred_neg)))
        regulation term = lambda * |initial_embedding|^2
        """
        # regulation term
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        # BPR loss
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.propagate()
        # get ui embeddings
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_prod = torch.mul(users_emb, items_emb)  # inner_product similarity between chosen ui
        gamma = torch.sum(inner_prod, dim=1)  # sum on user?
        return gamma


# device = config['device']
# dataset = Loader()
# model = LightGCN(config, dataset).to(device)
#
# input_user = torch.tensor([1])
# input_user = torch.tensor([1, 2, 3])

# print(model.graph.shape)
# users_emb = model.user_embedding.weight
# items_emb = model.item_embedding.weight
# all_emb = torch.cat([users_emb, items_emb])
# print('users_emb', users_emb.shape)
# print('items_emb', items_emb.shape)
# print(model.getUsersRating(input_user))

# print(
#     torch.topk(model.getUsersRating(input_user), 20).indices
# )
