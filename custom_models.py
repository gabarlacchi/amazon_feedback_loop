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
        
        return scores.view(-1)
    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain."""
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)


        user = interaction[self.USER_ID]
        
        user_all_embeddings, item_all_embeddings = self.forward()
        
        u_embeddings = user_all_embeddings[user]
        
        # Compute scores for all items
        scores = torch.matmul(u_embeddings, item_all_embeddings.transpose(0, 1))
        
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
    
class ItemKNN(GeneralRecommender):
    """
    ItemKNN is a basic model that computes item similarity with the interaction matrix.
    Can also be configured for user-based collaborative filtering.
    """
    
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL
    
    def __init__(self, config, dataset):
        super(ItemKNN, self).__init__(config, dataset)
        
        # Load parameters
        self.k = config["k"]
        self.method = "item"  # 'item' or 'user'
        self.shrink = config["shrink"]
        
        # Get interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        
        # Compute similarity matrix
        _, self.w = ComputeSimilarity(
            self.interaction_matrix,
            topk=self.k,
            shrink=self.shrink,
            method=self.method
        ).compute_similarity()
        
        # Pre-compute prediction matrix
        if self.method == "user":
            # User-based: W is user-user similarity
            self.pred_mat = self.w.dot(self.interaction_matrix).tolil()
        else:
            # Item-based: W is item-item similarity
            self.pred_mat = self.interaction_matrix.dot(self.w).tolil()
        
        # Fake loss for compatibility with RecBole trainer
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["w", "pred_mat"]
    
    def forward(self, user, item):
        """
        Forward pass - not used in traditional models but required by interface.
        """
        user = user.cpu().numpy()
        item = item.cpu().numpy()
        result = []
        
        for u, i in zip(user, item):
            result.append(self.pred_mat[u, i])
            
        return torch.FloatTensor(result).to(self.device)
    
    def calculate_loss(self, interaction):
        """
        Calculate loss - returns fake loss for traditional models.
        """
        return torch.nn.Parameter(torch.zeros(1)).to(self.device)
    
    def predict(self, interaction):
        """
        Predict scores for user-item pairs.
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user = user.cpu().numpy().astype(int)
        item = item.cpu().numpy().astype(int)
        
        result = []
        for u, i in zip(user, item):
            result.append(self.pred_mat[u, i])
            
        return torch.FloatTensor(result).to(self.device)
    
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for given users.
        """
        user = interaction[self.USER_ID]
        user = user.cpu().numpy().astype(int)
        
        score_list = []
        for u in user:
            # Get all scores for this user
            scores = self.pred_mat[u, :].toarray().flatten()
            score_list.append(scores)
        
        result = torch.FloatTensor(np.array(score_list)).to(self.device)
        return result.view(-1)

class UserKNN(GeneralRecommender):
    """
    UserKNN - User-based collaborative filtering.
    This is a convenience wrapper that sets knn_method='user'.
    """
    
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL
    
    def __init__(self, config, dataset):
        # Force user-based method
        config["knn_method"] = "user"
        
        # Load parameters
        self.k = config["k"]
        self.method = "user"
        self.shrink = config["shrink"]
        
        super(UserKNN, self).__init__(config, dataset)
        
        # Get interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        
        # Compute similarity matrix
        _, self.w = ComputeSimilarity(
            self.interaction_matrix,
            topk=self.k,
            shrink=self.shrink,
            method=self.method
        ).compute_similarity()
        
        # Pre-compute prediction matrix (user-based)
        self.pred_mat = self.w.dot(self.interaction_matrix).tolil()
        
        # Fake loss for compatibility
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["w", "pred_mat"]
    
    def forward(self, user, item):
        """Forward pass."""
        user = user.cpu().numpy()
        item = item.cpu().numpy()
        result = []
        
        for u, i in zip(user, item):
            result.append(self.pred_mat[u, i])
            
        return torch.FloatTensor(result).to(self.device)
    
    def calculate_loss(self, interaction):
        """Returns fake loss."""
        return torch.nn.Parameter(torch.zeros(1)).to(self.device)
    
    def predict(self, interaction):
        """Predict scores for user-item pairs."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user = user.cpu().numpy().astype(int)
        item = item.cpu().numpy().astype(int)
        
        result = []
        for u, i in zip(user, item):
            result.append(self.pred_mat[u, i])
            
        return torch.FloatTensor(result).to(self.device)
    
    def full_sort_predict(self, interaction):
        """Predict scores for all items."""
        user = interaction[self.USER_ID]
        user = user.cpu().numpy().astype(int)
        
        score_list = []
        for u in user:
            scores = self.pred_mat[u, :].toarray().flatten()
            score_list.append(scores)
        
        result = torch.FloatTensor(np.array(score_list)).to(self.device)
        return result.view(-1)