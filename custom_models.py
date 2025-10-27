from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.general_recommender.itemknn import ComputeSimilarity
from recbole.utils import InputType, ModelType
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_uniform_initialization
from recbole.data.interaction import Interaction
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from torch.nn.init import normal_

import torch.nn as nn
import random
import torch
import numpy as np
import scipy.sparse as sp


class UserKNN(GeneralRecommender):
    r"""ItemKNN is a basic model that compute item similarity with the interaction matrix."""
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(UserKNN, self).__init__(config, dataset)

        # load parameters info
        self.shrink = config["shrink"] if "shrink" in config else 0.0
        self.k = 50 # Neighbors

        self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        # print(f"\n Interaction matrix: {self.interaction_matrix.shape} \n")
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]

        _, self.w = ComputeSimilarity(
            self.interaction_matrix, topk=self.k, shrink=self.shrink
        ).compute_similarity("user")
        self.pred_mat = self.w.dot(self.interaction_matrix).tolil()

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["w", "pred_mat"]
        
    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user = user.cpu().numpy().astype(int)
        item = item.cpu().numpy().astype(int)
        result = []

        for index in range(len(user)):
            uid = user[index]
            iid = item[index]
            score = self.pred_mat[uid, iid]
            result.append(score)
        result = torch.from_numpy(np.array(result)).to(self.device)
        return result
    
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user = user.cpu().numpy()

        score = self.pred_mat[user, :].toarray().flatten()
        result = torch.from_numpy(score).to(self.device)

        return result
    
    def evaluate(self, test_interaction):
        users = torch.unique(test_interaction[self.USER_ID])

        k = 10 # numbero of prediction
        results = torch.tensor([]).to(self.device)
        for i in range(k):
            results = torch.cat([results, self.full_sort_predict(test_interaction)], dim = 1)
        
        # compute metrics and return a dict
        y = []
        for u in users:
            y.append(int(test_interaction[test_interaction[self.USER_ID] == u][self.ITEM_ID][-1]))
        
        hit_sum = 0 
        prec_sum = 0
        rec_sum = 0

        for i in range(len(y)):
            hit_sum += int(y[i] in results[i,:])
            intersection = len(set(results[i,:].numpy()).intersection(set([y[i]])))
            prec_sum +=  intersection / len([y[0]])
            rec_sum += intersection / len(results[i,:])

        results = {
            'hit@10': hit_sum/len(users),
            'precision@10': prec_sum/len(users),
            'recall@10': rec_sum/len(users)
        }

        return results

class IndividualRandom(GeneralRecommender):
    """Random is an fundamental model that recommends random items."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(IndividualRandom, self).__init__(config, dataset)
        self.dataset = dataset

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
    
    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        results = []

        for u in users:
            u = u.cpu()

            user_interaction_items = self.dataset.inter_matrix().toarray()[u, :]
            
            user_interacted_items = self.dataset.inter_feat[self.dataset.inter_feat[self.USER_ID] == u][self.ITEM_ID].numpy()
            user_interacted_items = list(set(list(user_interacted_items)))
            idxs = np.where(user_interaction_items > 0.0)[0]

            assert len(list(idxs)) == len(user_interacted_items)

            user_interaction_items[idxs] = 1/len(list(idxs))
            user_interaction_items[~idxs] = 0.0
            results.append(user_interaction_items)
        
        return torch.tensor(np.array(results)).reshape(1,-1)
    
    def evaluate(self, test_interaction):
        users = torch.unique(test_interaction[self.USER_ID])

        k = 10 # numbero of prediction
        results = torch.tensor([]).to(self.device)
        for i in range(k):
            results = torch.cat([results, self.full_sort_predict(test_interaction)], dim = 1)
        
        # compute metrics and return a dict
        y = []
        for u in users:
            y.append(int(test_interaction[test_interaction[self.USER_ID] == u][self.ITEM_ID][-1]))
        
        
        hit_sum = 0 
        prec_sum = 0
        rec_sum = 0

        for i in range(len(y)):
            hit_sum += int(y[i] in results[i,:])
            intersection = len(set(results[i,:].numpy()).intersection(set([y[i]])))
            prec_sum +=  intersection / len([y[0]])
            rec_sum += intersection / len(results[i,:])

        results = {
            'hit@10': hit_sum/len(users),
            'precision@10': prec_sum/len(users),
            'recall@10': rec_sum/len(users)
        }

        return results

class IndividualPopularity(GeneralRecommender):

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(IndividualPopularity, self).__init__(config, dataset)

        self.dataset = dataset

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
    
    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))
    
    def full_sort_predict(self, interaction):
        users = torch.unique(interaction[self.USER_ID])
        results = []
        for u in users:
            u = u.cpu()
            user_interaction_items = self.dataset.inter_matrix().toarray()[u, :]
            user_interaction_items = user_interaction_items / np.sum(user_interaction_items)
            
            results.append(user_interaction_items)
        
        return torch.tensor(np.array(results)).reshape(1,-1)
    
    def evaluate(self, dataset):
        users = torch.unique(dataset.inter_feat[self.USER_ID]).reshape(-1,1)

        k = 10 # number of prediction
        results = torch.tensor([])
        
        results = torch.cat([results, self.full_sort_predict(Interaction({self.USER_ID: users}))], dim = 1)
        
        # compute metrics and return a dict
        y = []
        for u in users:
            y.append(int(dataset.inter_feat[dataset.inter_feat[self.USER_ID] == u][self.ITEM_ID][-1]))
        
        
        hit_sum = 0 
        prec_sum = 0
        rec_sum = 0

        for i in range(len(y)):
            hit_sum += int(y[i] in results[i,:])
            intersection = len(set(results[i,:]).intersection(set([y[i]])))
            prec_sum +=  intersection / len([y[0]])
            rec_sum += intersection / len(results[i,:])

        results = {
            'hit@10': hit_sum/len(users),
            'precision@10': prec_sum/len(users),
            'recall@10': rec_sum/len(users)
        }

        return results

class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A = A.tolil()
        for (row, col), value in data_dict.items():
            A[row, col] = value
        A = A.todok()
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

class SpectralCF(GeneralRecommender):
    r"""SpectralCF is a spectral convolution model that directly learns latent factors of users and items 
    from the spectral domain for recommendation.

    The spectral convolution operation with C input channels and F filters is shown as the following:

    .. math::
        \left[\begin{array} {c} X_{new}^{u} \\
        X_{new}^{i} \end{array}\right]=\sigma\left(\left(U U^{\top}+U \Lambda U^{\top}\right)
        \left[\begin{array}{c} X^{u} \\
        X^{i} \end{array}\right] \Theta^{\prime}\right)

    where :math:`X_{new}^{u} \in R^{n_{users} \times F}` and :math:`X_{new}^{i} \in R^{n_{items} \times F}` 
    denote convolution results learned with F filters from the spectral domain for users and items, respectively; 
    :math:`\sigma` denotes the logistic sigmoid function.

    Note:

        Our implementation is a improved version which is different from the original paper.
        For a better stability, we replace :math:`U U^T` with identity matrix :math:`I` and
        replace :math:`U \Lambda U^T` with laplace matrix :math:`L`.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SpectralCF, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.emb_dim = config["embedding_size"]
        self.reg_weight = config["reg_weight"]

        # generate intermediate data
        # "A_hat = I + L" is equivalent to "A_hat = U U^T + U \Lambda U^T"
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        I = self.get_eye_mat(self.n_items + self.n_users)
        L = self.get_laplacian_matrix()
        A_hat = I + L
        self.A_hat = A_hat.to(self.device)

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.emb_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.emb_dim
        )
        self.filters = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.normal(
                        mean=0.01, std=0.02, size=(self.emb_dim, self.emb_dim)
                    ),
                    requires_grad=True,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.sigmoid = torch.nn.Sigmoid()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_item_e = None

        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_laplacian_matrix(self):
        r"""Get the laplacian matrix of users and items.

        .. math::
            L = I - D^{-1} \times A

        Returns:
            Sparse tensor of the laplacian matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        # A._update(data_dict)
        for (row, col), value in data_dict.items():
            A[row, col] = value

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -1)
        D = sp.diags(diag)
        A_tilde = D * A

        # covert norm_adj matrix to tensor
        A_tilde = sp.coo_matrix(A_tilde)
        row = A_tilde.row
        col = A_tilde.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(A_tilde.data)
        A_tilde = torch.sparse.FloatTensor(i, data, torch.Size(A_tilde.shape))

        # generate laplace matrix
        L = self.get_eye_mat(self.n_items + self.n_users) - A_tilde
        return L

    def get_eye_mat(self, num):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Args:
            num: number of column of the square matrix

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for k in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.A_hat, all_embeddings)
            all_embeddings = self.sigmoid(torch.mm(all_embeddings, self.filters[k]))
            embeddings_list.append(all_embeddings)

        new_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            new_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)

class BPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPR, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

        self.dataset = dataset
        # self.push_diversity = float(push_diversity)
        # self.user_seen_items = user_seen_items
        # if self.push_diversity > 0.0 and not len(self.user_seen_items):
        #     raise Exception(f"User seen items must not be empty")

    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        
        # if float(self.push_diversity) > 0.0:
        #     if random.random() < float(self.push_diversity):
        #         current_user_id = user.item()

        #         bought_item_indices = self.user_seen_items[current_user_id].to(score.device)
        #         all_item_indices = torch.arange(score.size(1), device=score.device)
        #         bought_mask = torch.isin(all_item_indices, bought_item_indices)

        #         score[:, bought_mask] *= 0.5  # Penalize bought items
        #         score[:, ~bought_mask] *= 2  # Boost unbought items

        return score.view(-1)

class NeuMF(GeneralRecommender):
    r"""NeuMF is an neural network enhanced matrix factorization model.
    It replace the dot product to mlp for a more precise user-item interaction.

    Note:

        Our implementation only contains a rough pretraining function.

    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NeuMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]

        # load parameters info
        self.mf_embedding_size = config["mf_embedding_size"]
        self.mlp_embedding_size = config["mlp_embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        self.mf_train = config["mf_train"]
        self.mlp_train = config["mlp_train"]
        self.use_pretrain = config["use_pretrain"]
        self.mf_pretrain_path = config["mf_pretrain_path"]
        self.mlp_pretrain_path = config["mlp_pretrain_path"]

        # define layers and loss
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        self.mlp_layers = MLPLayers(
            [2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob
        )
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(
                self.mf_embedding_size + self.mlp_hidden_size[-1], 1
            )
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)

    def load_pretrain(self):
        r"""A simple implementation of loading pretrained parameters."""
        mf = torch.load(self.mf_pretrain_path, map_location="cpu")
        mlp = torch.load(self.mlp_pretrain_path, map_location="cpu")
        mf = mf if "state_dict" not in mf else mf["state_dict"]
        mlp = mlp if "state_dict" not in mlp else mlp["state_dict"]
        self.user_mf_embedding.weight.data.copy_(mf["user_mf_embedding.weight"])
        self.item_mf_embedding.weight.data.copy_(mf["item_mf_embedding.weight"])
        self.user_mlp_embedding.weight.data.copy_(mlp["user_mlp_embedding.weight"])
        self.item_mlp_embedding.weight.data.copy_(mlp["item_mlp_embedding.weight"])

        mlp_layers = list(self.mlp_layers.state_dict().keys())
        index = 0
        for layer in self.mlp_layers.mlp_layers:
            if isinstance(layer, nn.Linear):
                weight_key = "mlp_layers." + mlp_layers[index]
                bias_key = "mlp_layers." + mlp_layers[index + 1]
                assert (
                    layer.weight.shape == mlp[weight_key].shape
                ), f"mlp layer parameter shape mismatch"
                assert (
                    layer.bias.shape == mlp[bias_key].shape
                ), f"mlp layer parameter shape mismatch"
                layer.weight.data.copy_(mlp[weight_key])
                layer.bias.data.copy_(mlp[bias_key])
                index += 2

        predict_weight = torch.cat(
            [mf["predict_layer.weight"], mlp["predict_layer.weight"]], dim=1
        )
        predict_bias = mf["predict_layer.bias"] + mlp["predict_layer.bias"]

        self.predict_layer.weight.data.copy_(predict_weight)
        self.predict_layer.bias.data.copy_(0.5 * predict_bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(
                torch.cat((user_mlp_e, item_mlp_e), -1)
            )  # [batch_size, layers[-1]]
        if self.mf_train and self.mlp_train:
            output = self.predict_layer(torch.cat((mf_output, mlp_output), -1))
        elif self.mf_train:
            output = self.predict_layer(mf_output)
        elif self.mlp_train:
            output = self.predict_layer(mlp_output)
        else:
            raise RuntimeError(
                "mf_train and mlp_train can not be False at the same time"
            )
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        predict = self.sigmoid(self.forward(user, item))
        return predict

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for a given user.
        
        Args:
            interaction (torch.Tensor): A tensor containing the user ID
        
        Returns:
            torch.Tensor: Predicted scores for all items for the given user
        """
        user = interaction[self.USER_ID]
        
        all_items = torch.arange(self.n_items, device=user.device).unsqueeze(0).repeat(user.size(0), 1)
        
        user = user.unsqueeze(1).repeat(1, self.n_items)
        
        user_flat = user.flatten()
        items_flat = all_items.flatten()
        
        with torch.no_grad():
            output = self.forward(user_flat, items_flat)
            
        scores = output.view(all_items.shape)

        # print(scores.view(-1))
        # s = scores.view(-1).cpu().numpy()
        # print(s)
        # print(scores.view(-1).size())

        # exit()
        # scores = self.sigmoid(scores)
        # scores = torch.sigmoid(scores)
        
        return scores.view(-1)
    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain."""
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)

class NNCF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NNCF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.ui_embedding_size = config["ui_embedding_size"]
        self.neigh_embedding_size = config["neigh_embedding_size"]
        self.num_conv_kernel = config["num_conv_kernel"]
        self.conv_kernel_size = config["conv_kernel_size"]
        self.pool_kernel_size = config["pool_kernel_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.neigh_num = config["neigh_num"]
        self.neigh_info_method = config["neigh_info_method"]
        self.resolution = config["resolution"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.ui_embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.ui_embedding_size)
        self.user_neigh_embedding = nn.Embedding(
            self.n_items, self.neigh_embedding_size
        )
        self.item_neigh_embedding = nn.Embedding(
            self.n_users, self.neigh_embedding_size
        )
        self.user_conv = nn.Sequential(
            nn.Conv1d(
                self.neigh_embedding_size, self.num_conv_kernel, self.conv_kernel_size
            ),
            nn.MaxPool1d(self.pool_kernel_size),
            nn.ReLU(),
        )
        self.item_conv = nn.Sequential(
            nn.Conv1d(
                self.neigh_embedding_size, self.num_conv_kernel, self.conv_kernel_size
            ),
            nn.MaxPool1d(self.pool_kernel_size),
            nn.ReLU(),
        )
        conved_size = self.neigh_num - (self.conv_kernel_size - 1)
        pooled_size = (
            conved_size - (self.pool_kernel_size - 1) - 1
        ) // self.pool_kernel_size + 1
        self.mlp_layers = MLPLayers(
            [2 * pooled_size * self.num_conv_kernel + self.ui_embedding_size]
            + self.mlp_hidden_size,
            config["dropout"],
        )
        self.out_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.dropout_layer = torch.nn.Dropout(p=config["dropout"])
        self.loss = nn.BCEWithLogitsLoss()

        # choose the method to use neighborhood information
        if self.neigh_info_method == "random":
            self.u_neigh, self.i_neigh = self.get_neigh_random()
        elif self.neigh_info_method == "knn":
            self.u_neigh, self.i_neigh = self.get_neigh_knn()
        elif self.neigh_info_method == "louvain":
            self.u_neigh, self.i_neigh = self.get_neigh_louvain()
        else:
            raise RuntimeError(
                "You need to choose the right algorithm of processing neighborhood information. \
                The parameter neigh_info_method can be set to random, knn or louvain."
            )

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    # Unify embedding length
    def Max_ner(self, lst, max_ner):
        r"""Unify embedding length of neighborhood information for efficiency consideration.
        Truncate the list if the length is larger than max_ner.
        Otherwise, pad it with 0.

        Args:
            lst (list): The input list contains node's neighbors.
            max_ner (int): The number of neighbors we choose for each node.

        Returns:
            list: The list of a node's community neighbors.


        """
        for i in range(len(lst)):
            if len(lst[i]) >= max_ner:
                lst[i] = lst[i][:max_ner]
            else:
                length = len(lst[i])
                for _ in range(max_ner - length):
                    lst[i].append(0)
        return lst


    # Find other nodes in the same community
    def get_community_member(self, partition, community_dict, node, kind):
        r"""Find other nodes in the same community.
        e.g. If the node starts with letter "i",
        the other nodes start with letter "i" in the same community dict group are its community neighbors.

        Args:
            partition (dict): The input dict that contains the community each node belongs.
            community_dict (dict): The input dict that shows the nodes each community contains.
            node (int): The id of the input node.
            kind (char): The type of the input node.

        Returns:
            list: The list of a node's community neighbors.

        """
        comm = community_dict[partition[node]]
        return [x for x in comm if x.startswith(kind)]


    # Prepare neiborhood embeddings, i.e. I(u) and U(i)
    def prepare_vector_element(self, partition, relation, community_dict):
        r"""Find the community neighbors of each node, i.e. I(u) and U(i).
        Then reset the id of nodes.

        Args:
            partition (dict): The input dict that contains the community each node belongs.
            relation (list): The input list that contains the relationships of users and items.
            community_dict (dict): The input dict that shows the nodes each community contains.

        Returns:
            list: The list of nodes' community neighbors.

        """
        item2user_neighbor_lst = [[] for _ in range(self.n_items)]
        user2item_neighbor_lst = [[] for _ in range(self.n_users)]

        for r in range(len(relation)):
            user, item = relation[r][0], relation[r][1]
            item2user_neighbor = self.get_community_member(
                partition, community_dict, user, "u"
            )
            np.random.shuffle(item2user_neighbor)
            user2item_neighbor = self.get_community_member(
                partition, community_dict, item, "i"
            )
            np.random.shuffle(user2item_neighbor)
            _, user = user.split("_", 1)
            user = int(user)
            _, item = item.split("_", 1)
            item = int(item)
            for i in range(len(item2user_neighbor)):
                name, index = item2user_neighbor[i].split("_", 1)
                item2user_neighbor[i] = int(index)
            for i in range(len(user2item_neighbor)):
                name, index = user2item_neighbor[i].split("_", 1)
                user2item_neighbor[i] = int(index)

            item2user_neighbor_lst[item] = item2user_neighbor
            user2item_neighbor_lst[user] = user2item_neighbor

        return user2item_neighbor_lst, item2user_neighbor_lst


    # Get neighborhood embeddings using louvain method
    def get_neigh_louvain(self):
        r"""Get neighborhood information using louvain algorithm.
        First, change the id of node,
        for example, the id of user node "1" will be set to "u_1" in order to use louvain algorithm.
        Second, use louvain algorithm to seperate nodes into different communities.
        Finally, find the community neighbors of each node with the same type and reset the id of the nodes.

        Returns:
            torch.IntTensor: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
        """
        inter_M = self.interaction_matrix
        pairs = list(zip(inter_M.row, inter_M.col))

        tmp_relation = []
        for i in range(len(pairs)):
            tmp_relation.append(
                ["user_" + str(pairs[i][0]), "item_" + str(pairs[i][1])]
            )

        import networkx as nx

        G = nx.Graph()
        G.add_edges_from(tmp_relation)
        resolution = self.resolution
        import community

        partition = community.best_partition(G, resolution=resolution)

        community_dict = {}
        community_dict.setdefault(0, [])
        for i in range(len(partition.values())):
            community_dict[i] = []
        for node, part in partition.items():
            community_dict[part] = community_dict[part] + [node]

        tmp_user2item, tmp_item2user = self.prepare_vector_element(
            partition, tmp_relation, community_dict
        )
        u_neigh = self.Max_ner(tmp_user2item, self.neigh_num)
        i_neigh = self.Max_ner(tmp_item2user, self.neigh_num)

        u_neigh = torch.tensor(u_neigh, device=self.device)
        i_neigh = torch.tensor(i_neigh, device=self.device)
        return u_neigh, i_neigh


    # Get neighborhood embeddings using knn method
    def get_neigh_knn(self):
        r"""Get neighborhood information using knn algorithm.
        Find direct neighbors of each node, if the number of direct neighbors is less than neigh_num,
        add other similar neighbors using knn algorithm.
        Otherwise, select random top k direct neighbors, k equals to the number of neighbors.

        Returns:
            torch.IntTensor: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
        """
        inter_M = self.interaction_matrix
        pairs = list(zip(inter_M.row, inter_M.col))
        ui_inters = np.zeros((self.n_users, self.n_items), dtype=np.int8)

        for i in range(len(pairs)):
            ui_inters[pairs[i][0], pairs[i][1]] = 1

        # Get similar neighbors using knn algorithm
        user_knn, _ = ComputeSimilarity(
            self.interaction_matrix.tocsr(), topk=self.neigh_num
        ).compute_similarity("user")
        item_knn, _ = ComputeSimilarity(
            self.interaction_matrix.tocsr(), topk=self.neigh_num
        ).compute_similarity("item")

        u_neigh, i_neigh = [], []

        for u in range(self.n_users):
            neigh_list = ui_inters[u].nonzero()[0]
            direct_neigh_num = len(neigh_list)
            if len(neigh_list) == 0:
                u_neigh.append(self.neigh_num * [0])
            elif direct_neigh_num < self.neigh_num:
                tmp_k = self.neigh_num - direct_neigh_num
                mask = np.random.randint(0, len(neigh_list), size=1)
                neigh_list = list(neigh_list) + list(item_knn[neigh_list[mask[0]]])
                u_neigh.append(neigh_list[: self.neigh_num])
            else:
                mask = np.random.randint(0, len(neigh_list), size=self.neigh_num)
                u_neigh.append(neigh_list[mask])

        for i in range(self.n_items):
            neigh_list = ui_inters[:, i].nonzero()[0]
            direct_neigh_num = len(neigh_list)
            if len(neigh_list) == 0:
                i_neigh.append(self.neigh_num * [0])
            elif direct_neigh_num < self.neigh_num:
                tmp_k = self.neigh_num - direct_neigh_num
                mask = np.random.randint(0, len(neigh_list), size=1)
                neigh_list = list(neigh_list) + list(user_knn[neigh_list[mask[0]]])
                i_neigh.append(neigh_list[: self.neigh_num])
            else:
                mask = np.random.randint(0, len(neigh_list), size=self.neigh_num)
                i_neigh.append(neigh_list[mask])

        u_neigh = torch.tensor(u_neigh, device=self.device)
        i_neigh = torch.tensor(i_neigh, device=self.device)
        return u_neigh, i_neigh


    # Get neighborhood embeddings using random method
    def get_neigh_random(self):
        r"""Get neighborhood information using random algorithm.
        Select random top k direct neighbors, k equals to the number of neighbors.

        Returns:
            torch.IntTensor: The neighborhood nodes of a batch of user or item, shape: [batch_size, neigh_num]
        """
        inter_M = self.interaction_matrix
        pairs = list(zip(inter_M.row, inter_M.col))
        ui_inters = np.zeros((self.n_users, self.n_items), dtype=np.int8)

        for i in range(len(pairs)):
            ui_inters[pairs[i][0], pairs[i][1]] = 1

        u_neigh, i_neigh = [], []

        for u in range(self.n_users):
            neigh_list = ui_inters[u].nonzero()[0]
            if len(neigh_list) == 0:
                u_neigh.append(self.neigh_num * [0])
            else:
                mask = np.random.randint(0, len(neigh_list), size=self.neigh_num)
                u_neigh.append(neigh_list[mask])

        for i in range(self.n_items):
            neigh_list = ui_inters[:, i].nonzero()[0]
            if len(neigh_list) == 0:
                i_neigh.append(self.neigh_num * [0])
            else:
                mask = np.random.randint(0, len(neigh_list), size=self.neigh_num)
                i_neigh.append(neigh_list[mask])

        u_neigh = torch.tensor(np.array(u_neigh), device=self.device)
        i_neigh = torch.tensor(np.array(i_neigh), device=self.device)
        return u_neigh, i_neigh


    # Get neighborhood embeddings
    def get_neigh_info(self, user, item):
        r"""Get a batch of neighborhood embedding tensor according to input id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The neighborhood embedding tensor of a batch of user, shape: [batch_size, neigh_embedding_size]
            torch.FloatTensor: The neighborhood embedding tensor of a batch of item, shape: [batch_size, neigh_embedding_size]

        """
        batch_u_neigh = self.u_neigh[user]
        batch_i_neigh = self.i_neigh[item]
        return batch_u_neigh, batch_i_neigh

    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)

        user_neigh_input, item_neigh_input = self.get_neigh_info(user, item)
        user_neigh_embedding = self.user_neigh_embedding(user_neigh_input)
        item_neigh_embedding = self.item_neigh_embedding(item_neigh_input)
        user_neigh_embedding = user_neigh_embedding.permute(0, 2, 1)
        user_neigh_conv_embedding = self.user_conv(user_neigh_embedding)
        # batch_size * out_channel * pool_size
        batch_size = user_neigh_conv_embedding.size(0)
        user_neigh_conv_embedding = user_neigh_conv_embedding.view(batch_size, -1)
        item_neigh_embedding = item_neigh_embedding.permute(0, 2, 1)
        item_neigh_conv_embedding = self.item_conv(item_neigh_embedding)
        # batch_size * out_channel * pool_size
        item_neigh_conv_embedding = item_neigh_conv_embedding.view(batch_size, -1)
        mf_vec = torch.mul(user_embedding, item_embedding)
        last = torch.cat(
            (mf_vec, user_neigh_conv_embedding, item_neigh_conv_embedding), dim=-1
        )

        output = self.mlp_layers(last)
        out = self.out_layer(output)
        out = out.squeeze(-1)
        return out

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item)
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return torch.sigmoid(self.forward(user, item))

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        
        # Get all item IDs
        all_items = torch.arange(self.n_items).to(self.device)
        
        # Create user-item pairs for all items
        user = user.unsqueeze(1).expand(-1, self.n_items)
        all_items = all_items.unsqueeze(0).expand(len(user), -1)
        
        # Flatten to create pairs
        user = user.flatten()
        all_items = all_items.flatten()
        
        # Create a new interaction dict with these pairs
        new_interaction = {
            self.USER_ID: user,
            self.ITEM_ID: all_items
        }
        
        # Use the existing predict function
        scores = self.predict(new_interaction)

        scores = scores.view(all_items.shape)
        
        # Reshape scores to [num_users, num_items]
        return scores.view(-1)