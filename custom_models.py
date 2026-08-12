from recbole.model.abstract_recommender import GeneralRecommender, ContextRecommender
from recbole.model.general_recommender.itemknn import ComputeSimilarity
from recbole.utils import InputType, ModelType
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_uniform_initialization
from recbole.data.interaction import Interaction
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers, BaseFactorizationMachine
from torch.nn.init import normal_
from recbole.model.general_recommender.lightgcn import LightGCN
import math


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

class FM(ContextRecommender):
    
    def __init__(self, config, dataset):
        super(FM, self).__init__(config, dataset)
        
        self.dataset = dataset
        self.USER_ID = dataset.uid_field
        self.ITEM_ID = dataset.iid_field
        self.LABEL = dataset.label_field
        
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_users = dataset.num(self.USER_ID)
        self.embedding_size = config['embedding_size']
        
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        
        # FIXED: Get the actual number of fields from the embedding field names
        # This includes user_id, item_id, and all other features
        self.num_feature_field = len(self.token_field_names) + len(self.float_field_names)
        
        # If token/float fields don't include user_id and item_id, add them
        if self.USER_ID not in self.token_field_names:
            self.num_feature_field += 1
        if self.ITEM_ID not in self.token_field_names:
            self.num_feature_field += 1
        
        self.first_order_linear = nn.Embedding(self.num_feature_field, 1)
        nn.init.normal_(self.first_order_linear.weight, mean=0, std=0.01)
        
        self.bias = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, interaction):
        # [batch_size, num_field, embed_dim]
        fm_all_embeddings = self.concat_embed_input_fields(interaction)

        # DYNAMIC: Get actual number of fields from the embeddings
        actual_num_fields = fm_all_embeddings.size(1)
        
        # First order - use only the fields that actually exist
        first_order = self.first_order_linear.weight[:actual_num_fields].squeeze(1)  # [actual_num_fields]
        first_order_output = torch.sum(
            fm_all_embeddings * first_order.unsqueeze(0).unsqueeze(-1),
            dim=(1, 2)
        )  # [batch_size]
        
        # Second order (FM interaction)
        second_order_output = self.fm(fm_all_embeddings)  # [batch_size] or [batch_size, 1]
        
        # CRITICAL FIX: Ensure second_order is 1D
        if second_order_output.dim() > 1:
            second_order_output = second_order_output.squeeze(-1)
        
        # Combine
        output = first_order_output + second_order_output + self.bias.squeeze()
        return output

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        output = self.forward(interaction)
        return self.sigmoid(output)

    @torch.no_grad()
    def full_sort_predict(self, interaction):

        device = interaction[self.USER_ID].device
        batch_size = interaction[self.USER_ID].size(0)
        n_items = self.n_items
        
        # Join user features FIRST
        interaction = self.dataset.join(interaction)
        
        # Update batch_size after join
        batch_size = interaction[self.USER_ID].size(0)
        
        # Expand for all items
        item_ids = torch.arange(n_items, device=device).unsqueeze(0).repeat(batch_size, 1)
        item_ids_flat = item_ids.reshape(-1)

        new_inter_dict = {}

        for key in interaction.interaction:
            if key == self.ITEM_ID:
                continue
            v = interaction[key]
            reps = [n_items] + [1] * (v.dim() - 1)
            new_inter_dict[key] = v.repeat(*reps).reshape(batch_size * n_items, *v.shape[1:])

        if self.ITEM_ID in interaction.interaction:
            dtype = interaction[self.ITEM_ID].dtype
        else:
            dtype = torch.long
        new_inter_dict[self.ITEM_ID] = item_ids_flat.to(dtype=dtype)

        full_inter = Interaction(new_inter_dict).to(device)
        
        # Join item features
        full_inter = self.dataset.join(full_inter)
        
        # Forward pass
        logits_flat = self.forward(full_inter)
        
        scores_flat = self.sigmoid(logits_flat)

        # Reshape
        scores = scores_flat.view(batch_size, n_items)
        
        if batch_size == 1:
            scores = scores.squeeze(0)
        
        return scores

class CrossNetworkV2(nn.Module):


    """Cross Network V2 with mixture of experts."""
    
    def __init__(self, input_dim, num_layers, low_rank=32, num_experts=4):
        super(CrossNetworkV2, self).__init__()
        self.num_layers = num_layers
        self.low_rank = low_rank
        self.num_experts = num_experts
        
        # Mixture of experts for each layer
        self.expert_weights = nn.ModuleList([
            nn.Linear(input_dim, low_rank * num_experts, bias=False)
            for _ in range(num_layers)
        ])
        
        self.expert_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim, num_experts))
            for _ in range(num_layers)
        ])
        
        self.gate = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False)
            for _ in range(num_layers)
        ])
        
        # V matrices for low-rank approximation
        self.v_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, low_rank))
            for _ in range(num_layers)
        ])
        
        # Initialize weights
        for i in range(num_layers):
            nn.init.xavier_normal_(self.expert_weights[i].weight)
            nn.init.xavier_normal_(self.v_weights[i])
            nn.init.zeros_(self.gate[i].weight)
    
    def forward(self, x0):
        """
        Args:
            x0: [batch_size, input_dim]
        Returns:
            x: [batch_size, input_dim]
        """
        x = x0
        for i in range(self.num_layers):
            # Expert outputs: [batch_size, low_rank * num_experts]
            expert_out = self.expert_weights[i](x)
            
            # Reshape to [batch_size, low_rank, num_experts]
            expert_out = expert_out.view(-1, self.low_rank, self.num_experts)
            
            # Gate: [batch_size, 1]
            gate_out = self.gate[i](x)
            gate_out = torch.softmax(gate_out, dim=-1)  # [batch_size, 1]
            
            # Mix experts: [batch_size, low_rank, num_experts] @ [batch_size, num_experts, 1]
            # -> [batch_size, low_rank, 1] -> [batch_size, low_rank]
            gate_out_expanded = gate_out.unsqueeze(1)  # [batch_size, 1, 1]
            gated_expert = (expert_out * gate_out_expanded).sum(dim=2)  # [batch_size, low_rank]
            
            # Low-rank transformation: [batch_size, low_rank] @ [low_rank, input_dim]^T
            # -> [batch_size, input_dim]
            cross_out = torch.matmul(gated_expert, self.v_weights[i].t())
            
            # Add bias
            bias_out = (self.expert_bias[i] * gate_out_expanded).sum(dim=2)  # [batch_size, input_dim]
            cross_out = cross_out + bias_out
            
            # Element-wise product with x0 and add x
            x = x0 * cross_out + x
        
        return x

class DCNV2(ContextRecommender):
    """Deep & Cross Network V2 for CTR prediction."""
    
    input_type = InputType.POINTWISE
    
    def __init__(self, config, dataset):
        super(DCNV2, self).__init__(config, dataset)
        
        # Keep dataset reference for join() in full_sort_predict
        self.dataset = dataset
        self.USER_ID = dataset.uid_field
        self.ITEM_ID = dataset.iid_field
        self.LABEL = dataset.label_field
        
        # Cache item/user counts
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_users = dataset.num(self.USER_ID)
        
        # Get embedding size from config
        self.embedding_size = config['embedding_size']
        
        # Number of feature fields
        self.num_feature_field = len(self.token_field_names) + len(self.float_field_names)
        
        # Input dimension for cross and deep networks
        self.input_dim = self.num_feature_field * self.embedding_size

        DEFAULT_CONFIG = {
            'cross_layer_num': 3,
            'low_rank': 32,
            'num_experts': 4,
            'mlp_hidden_size': [256, 128, 64],
            'reg_weight': 1e-5,
            'structure': 'parallel'
        }
        
        # Cross Network V2 parameters
        self.cross_layer_num = config['cross_layer_num'] if 'cross_layer_num' in config else DEFAULT_CONFIG['cross_layer_num']
        self.low_rank = config['low_rank'] if 'low_rank' in config else DEFAULT_CONFIG['low_rank']
        self.num_experts = config['num_experts'] if 'num_experts' in config else DEFAULT_CONFIG['num_experts']
        self.mlp_hidden_size = config['mlp_hidden_size'] if 'mlp_hidden_size' in config else DEFAULT_CONFIG['mlp_hidden_size']
        self.reg_weight = config['reg_weight'] if 'reg_weight' in config else DEFAULT_CONFIG['reg_weight']
        self.structure = config['structure'] if 'structure' in config else DEFAULT_CONFIG['structure']
        self.dropout_prob = config['dropout_prob'] if 'dropout_prob' in config else 0.2
        
        # Cross Network V2
        self.cross_network = CrossNetworkV2(
            input_dim=self.input_dim,
            num_layers=self.cross_layer_num,
            low_rank=self.low_rank,
            num_experts=self.num_experts
        )
        
        # Deep Network (MLP)
        if self.structure == 'parallel':
            # Parallel structure: cross and deep networks are parallel
            self.mlp = MLPLayers(
                [self.input_dim] + self.mlp_hidden_size,
                self.dropout_prob,
                activation='relu',
                bn=True
            )
            # Combine cross and deep outputs
            combine_dim = self.input_dim + self.mlp_hidden_size[-1]
        else:
            # Stacked structure: deep network on top of cross network
            self.mlp = MLPLayers(
                [self.input_dim] + self.mlp_hidden_size,
                self.dropout_prob,
                activation='relu',
                bn=True
            )
            combine_dim = self.mlp_hidden_size[-1]
        
        # Final prediction layer
        self.predict_layer = nn.Linear(combine_dim, 1)
        
        # Activation and loss
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        
        # Initialize prediction layer
        nn.init.xavier_normal_(self.predict_layer.weight)
        nn.init.zeros_(self.predict_layer.bias)
    
    def forward(self, interaction):
        """
        Args:
            interaction: Interaction object
        Returns:
            output: [batch_size] prediction scores (logits)
        """
        # Get embeddings: [batch_size, num_field, embed_dim]
        embed_input = self.concat_embed_input_fields(interaction)
        batch_size = embed_input.size(0)
        
        # Flatten embeddings: [batch_size, num_field * embed_dim]
        flat_input = embed_input.view(batch_size, -1)
        
        if self.structure == 'parallel':
            # Parallel structure
            # Cross network output
            cross_out = self.cross_network(flat_input)  # [batch_size, input_dim]
            
            # Deep network output
            deep_out = self.mlp(flat_input)  # [batch_size, mlp_hidden_size[-1]]
            
            # Concatenate
            combined = torch.cat([cross_out, deep_out], dim=1)
        else:
            # Stacked structure
            # Cross network output
            cross_out = self.cross_network(flat_input)  # [batch_size, input_dim]
            
            # Deep network on top of cross output
            deep_out = self.mlp(cross_out)  # [batch_size, mlp_hidden_size[-1]]
            
            combined = deep_out
        
        # Final prediction
        output = self.predict_layer(combined).squeeze(-1)  # [batch_size]
        
        return output
    
    def calculate_loss(self, interaction):
        """Calculate loss for training."""
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        
        # BCE loss
        bce_loss = self.loss(output, label)
        
        # L2 regularization
        l2_loss = self.reg_weight * sum(
            torch.sum(param ** 2) for param in self.parameters()
        )
        
        return bce_loss + l2_loss
    
    def predict(self, interaction):
        """Predict for evaluation."""
        output = self.forward(interaction)
        return self.sigmoid(output)
    
    @torch.no_grad()
    def full_sort_predict(self, interaction):
        """
        Full sort prediction for ranking all items.
        
        Args:
            interaction: User interaction with user features
        Returns:
            scores: [batch_size, n_items] or [n_items] if batch_size=1
        """
        device = interaction[self.USER_ID].device
        batch_size = interaction[self.USER_ID].size(0)
        n_items = self.n_items
        
        # Join user features FIRST
        interaction = self.dataset.join(interaction)
        
        # Update batch_size after join
        batch_size = interaction[self.USER_ID].size(0)
        
        # Expand for all items
        item_ids = torch.arange(n_items, device=device).unsqueeze(0).repeat(batch_size, 1)
        item_ids_flat = item_ids.reshape(-1)
        
        # Create new interaction with all user-item pairs
        new_inter_dict = {}
        for key in interaction.interaction:
            if key == self.ITEM_ID:
                continue
            v = interaction[key]
            reps = [n_items] + [1] * (v.dim() - 1)
            new_inter_dict[key] = v.repeat(*reps).reshape(batch_size * n_items, *v.shape[1:])
        
        # Add item IDs
        if self.ITEM_ID in interaction.interaction:
            dtype = interaction[self.ITEM_ID].dtype
        else:
            dtype = torch.long
        
        new_inter_dict[self.ITEM_ID] = item_ids_flat.to(dtype=dtype)
        full_inter = Interaction(new_inter_dict).to(device)
        
        # Join item features
        full_inter = self.dataset.join(full_inter)
        
        # Forward pass
        logits_flat = self.forward(full_inter)
        scores_flat = self.sigmoid(logits_flat)
        
        # Reshape to [batch_size, n_items]
        scores = scores_flat.view(batch_size, n_items)
        
        # If single user, squeeze batch dimension
        if batch_size == 1:
            scores = scores.squeeze(0)
        
        return scores

class DeepFM(ContextRecommender):
    def __init__(self, config, dataset):
        super(DeepFM, self).__init__(config, dataset)
        # Keep dataset reference for join() in full_sort_predict
        self.dataset = dataset
        self.USER_ID = dataset.uid_field
        self.ITEM_ID = dataset.iid_field
        self.LABEL = dataset.label_field
        # Cache item/user counts
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_users = dataset.num(self.USER_ID)
        # Get embedding size from config
        self.embedding_size = config['embedding_size']
        
        # FM components
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        self.num_feature_field = len(self.token_field_names) + len(self.float_field_names)
        
        # First order linear layer (per-field scalar)
        self.first_order_linear = nn.Embedding(self.num_feature_field, 1)
        nn.init.normal_(self.first_order_linear.weight, mean=0, std=0.01)
        
        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Deep component (MLP) - using RecBole MLPLayers
        self.mlp_hidden_size = config['mlp_hidden_size']  # list like [256, 128, 64]
        self.dropout_prob = config['dropout_prob']  # float like 0.2
        
        # Input to MLP is flattened embeddings
        input_size = self.num_feature_field * self.embedding_size
        
        # Use RecBole's MLPLayers if available, otherwise build manually
        try:
            from recbole.model.layers import MLPLayers
            self.mlp = MLPLayers(
                [input_size] + self.mlp_hidden_size,
                self.dropout_prob,
                activation='relu',
                bn=True
            )
            self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        except ImportError:
            # Manual implementation if MLPLayers not available
            mlp_layers = []
            for hidden_size in self.mlp_hidden_size:
                mlp_layers.append(nn.Linear(input_size, hidden_size))
                mlp_layers.append(nn.BatchNorm1d(hidden_size))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(self.dropout_prob))
                input_size = hidden_size
            self.mlp = nn.Sequential(*mlp_layers)
            self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        
        # Sigmoid for output
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, interaction):
        # [batch_size, num_field, embed_dim]
        fm_all_embeddings = self.concat_embed_input_fields(interaction)
        batch_size = fm_all_embeddings.size(0)
        
        # First order
        first_order = self.first_order_linear.weight.squeeze(1)  # [num_field]
        first_order_output = torch.sum(
            fm_all_embeddings * first_order.unsqueeze(0).unsqueeze(-1),
            dim=(1, 2)
        )  # [batch_size]
        
        # Second order (FM interaction)
        second_order_output = self.fm(fm_all_embeddings)  # [batch_size] or [batch_size, 1]
        
        # CRITICAL FIX: Ensure second_order is 1D
        if second_order_output.dim() > 1:
            second_order_output = second_order_output.squeeze(-1)
        
        # Deep component
        # Flatten embeddings for MLP input
        deep_input = fm_all_embeddings.view(batch_size, -1)  # [batch_size, num_field * embed_dim]
        deep_output = self.mlp(deep_input)  # [batch_size, last_hidden_size]
        deep_output = self.deep_predict_layer(deep_output).squeeze(-1)  # [batch_size]
        
        # Combine all components: y = w0 + first_order + second_order + deep
        output = self.bias.squeeze() + first_order_output + second_order_output + deep_output
        
        return output
    
    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)
    
    def predict(self, interaction):
        output = self.forward(interaction)
        return self.sigmoid(output)
    
    @torch.no_grad()
    def full_sort_predict(self, interaction):
        device = interaction[self.USER_ID].device
        batch_size = interaction[self.USER_ID].size(0)
        n_items = self.n_items
        
        # Join user features FIRST
        interaction = self.dataset.join(interaction)
        
        # Update batch_size after join
        batch_size = interaction[self.USER_ID].size(0)
        
        # Expand for all items
        item_ids = torch.arange(n_items, device=device).unsqueeze(0).repeat(batch_size, 1)
        item_ids_flat = item_ids.reshape(-1)
        
        new_inter_dict = {}
        for key in interaction.interaction:
            if key == self.ITEM_ID:
                continue
            v = interaction[key]
            reps = [n_items] + [1] * (v.dim() - 1)
            new_inter_dict[key] = v.repeat(*reps).reshape(batch_size * n_items, *v.shape[1:])
        
        if self.ITEM_ID in interaction.interaction:
            dtype = interaction[self.ITEM_ID].dtype
        else:
            dtype = torch.long
        
        new_inter_dict[self.ITEM_ID] = item_ids_flat.to(dtype=dtype)
        full_inter = Interaction(new_inter_dict).to(device)
        
        # Join item features
        full_inter = self.dataset.join(full_inter)
        
        # Forward pass
        logits_flat = self.forward(full_inter)
        scores_flat = self.sigmoid(logits_flat)
        
        # Reshape
        scores = scores_flat.view(batch_size, n_items)
        
        if batch_size == 1:
            scores = scores.squeeze(0)
        
        return scores

class NFM(ContextRecommender):
    """
    Neural Factorization Machine for Recommendation
    
    Combines:
    - Linear part (1st order)
    - Bi-Interaction pooling (2nd order, like FM)
    - Deep neural network on interactions
    
    Reference:
        He and Chua. "Neural Factorization Machines for Sparse Predictive Analytics" SIGIR 2017
    
    Simpler than xDeepFM, more powerful than FM/DeepFM
    """
    
    def __init__(self, config, dataset):
        super(NFM, self).__init__(config, dataset)
        
        # Keep dataset reference for join() in full_sort_predict
        self.dataset = dataset
        self.USER_ID = dataset.uid_field
        self.ITEM_ID = dataset.iid_field
        self.LABEL = dataset.label_field
        
        # Cache item/user counts
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_users = dataset.num(self.USER_ID)
        
        # Get config parameters
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size'] if 'mlp_hidden_size' in config else [128, 64]
        self.dropout = config['dropout_prob'] if 'dropout_prob' in config else 0.1
        
        # Number of feature fields
        self.num_feature_field = len(self.token_field_names) + len(self.float_field_names)
        
        # === Linear Part (First Order) ===
        self.first_order_linear = nn.Embedding(self.num_feature_field, 1)
        nn.init.normal_(self.first_order_linear.weight, mean=0, std=0.01)
        
        # === Bi-Interaction Layer ===
        # This is the key innovation: element-wise product pooling
        # No parameters needed, just operations on embeddings
        
        # === Deep Neural Network ===
        # Input: embedding_size (after bi-interaction pooling)
        # Output: final hidden layer
        self.dnn_layers = nn.ModuleList()
        prev_dim = self.embedding_size
        
        for hidden_dim in self.mlp_hidden_size:
            self.dnn_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.dnn_layers.append(nn.BatchNorm1d(hidden_dim))
            self.dnn_layers.append(nn.ReLU())
            self.dnn_layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        # === Final Prediction Layer ===
        # Combine linear + DNN output
        self.prediction_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        
        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Output activation and loss
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
    
    def bi_interaction_pooling(self, embeddings):
        """
        Bi-Interaction Pooling Layer
        
        Captures 2nd-order feature interactions efficiently:
        sum(vi * vj) for all i < j
        
        Equivalent to: 0.5 * (sum(vi)^2 - sum(vi^2))
        
        Args:
            embeddings: [batch_size, num_fields, embed_dim]
            
        Returns:
            pooled: [batch_size, embed_dim]
        """
        # Sum of embeddings: [batch_size, embed_dim]
        sum_of_embeddings = torch.sum(embeddings, dim=1)
        
        # Sum of squared embeddings: [batch_size, embed_dim]
        sum_of_squared_embeddings = torch.sum(embeddings ** 2, dim=1)
        
        # Square of sum: [batch_size, embed_dim]
        square_of_sum = sum_of_embeddings ** 2
        
        # Bi-interaction: 0.5 * (square_of_sum - sum_of_squares)
        bi_interaction = 0.5 * (square_of_sum - sum_of_squared_embeddings)
        
        return bi_interaction
    
    def forward(self, interaction):
        """
        Forward pass
        
        Args:
            interaction: Interaction object with user/item/context features
            
        Returns:
            output: [batch_size] logits
        """
        # Get embeddings: [batch_size, num_fields, embed_dim]
        nfm_all_embeddings = self.concat_embed_input_fields(interaction)
        batch_size = nfm_all_embeddings.shape[0]
        
        # === 1. Linear Part (First Order) ===
        first_order = self.first_order_linear.weight.squeeze(1)  # [num_fields]
        linear_output = torch.sum(
            nfm_all_embeddings * first_order.unsqueeze(0).unsqueeze(-1),
            dim=(1, 2)
        )  # [batch_size]
        
        # === 2. Bi-Interaction Pooling (Second Order) ===
        bi_output = self.bi_interaction_pooling(nfm_all_embeddings)  # [batch_size, embed_dim]
        
        # === 3. Deep Neural Network ===
        dnn_output = bi_output
        for layer in self.dnn_layers:
            dnn_output = layer(dnn_output)
        # dnn_output: [batch_size, mlp_hidden_size[-1]]
        
        # === 4. Final Prediction ===
        dnn_prediction = self.prediction_layer(dnn_output).squeeze(1)  # [batch_size]
        
        # Combine linear + DNN + bias
        output = linear_output + dnn_prediction + self.bias.squeeze()
        
        return output  # [batch_size]
    
    def calculate_loss(self, interaction):
        """Calculate BCE loss"""
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)
    
    def predict(self, interaction):
        """Predict probability for given interaction"""
        output = self.forward(interaction)
        return self.sigmoid(output)
    
    @torch.no_grad()
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for given users
        
        Args:
            interaction: Interaction with user features
            
        Returns:
            scores: [batch_size, n_items] or [n_items] if batch_size=1
        """
        device = interaction[self.USER_ID].device
        batch_size = interaction[self.USER_ID].size(0)
        n_items = self.n_items
        
        # Join user features FIRST
        interaction = self.dataset.join(interaction)
        
        # Update batch_size after join
        batch_size = interaction[self.USER_ID].size(0)
        
        # Expand for all items
        item_ids = torch.arange(n_items, device=device).unsqueeze(0).repeat(batch_size, 1)
        item_ids_flat = item_ids.reshape(-1)
        
        # Create full interaction matrix
        new_inter_dict = {}
        for key in interaction.interaction:
            if key == self.ITEM_ID:
                continue
            v = interaction[key]
            reps = [n_items] + [1] * (v.dim() - 1)
            new_inter_dict[key] = v.repeat(*reps).reshape(batch_size * n_items, *v.shape[1:])
        
        # Add item IDs
        if self.ITEM_ID in interaction.interaction:
            dtype = interaction[self.ITEM_ID].dtype
        else:
            dtype = torch.long
        new_inter_dict[self.ITEM_ID] = item_ids_flat.to(dtype=dtype)
        
        full_inter = Interaction(new_inter_dict).to(device)
        
        # Join item features
        full_inter = self.dataset.join(full_inter)
        
        # Forward pass
        logits_flat = self.forward(full_inter)
        scores_flat = self.sigmoid(logits_flat)
        
        # Reshape to [batch_size, n_items]
        scores = scores_flat.view(batch_size, n_items)
        
        if batch_size == 1:
            scores = scores.squeeze(0)
        
        return scores
    
class ComputeSimilarity:
    def __init__(self, dataMatrix, topk=100, shrink=0, method="item", normalize=True):
        r"""Computes the cosine similarity of dataMatrix

        If it is computed on :math:`URM=|users| \times |items|`, pass the URM.

        If it is computed on :math:`ICM=|items| \times |features|`, pass the ICM transposed.

        Args:
            dataMatrix (scipy.sparse.csr_matrix): The sparse data matrix.
            topk (int) : The k value in KNN.
            shrink (int) :  hyper-parameter in calculate cosine distance.
            method (str) : Calculate the similarity of users if method is 'user', otherwise, calculate the similarity of items.
            normalize (bool):   If True divide the dot product by the product of the norms.
        """

        super(ComputeSimilarity, self).__init__()

        self.shrink = shrink
        self.normalize = normalize
        self.method = method

        self.n_rows, self.n_columns = dataMatrix.shape

        if self.method == "user":
            self.TopK = min(topk, self.n_rows)
        else:
            self.TopK = min(topk, self.n_columns)

        self.dataMatrix = dataMatrix.copy()

    def compute_similarity(self, block_size=100):
        r"""Compute the similarity for the given dataset

        Args:
            block_size (int): divide matrix to :math:`n\_rows \div block\_size` to calculate cosine_distance if method is 'user',
                 otherwise, divide matrix to :math:`n\_columns \div block\_size`.

        Returns:

            list: The similar nodes, if method is 'user', the shape is [number of users, neigh_num],
            else, the shape is [number of items, neigh_num].
            scipy.sparse.csr_matrix: sparse matrix W, if method is 'user', the shape is [self.n_rows, self.n_rows],
            else, the shape is [self.n_columns, self.n_columns].
        """

        values = []
        rows = []
        cols = []
        neigh = []

        self.dataMatrix = self.dataMatrix.astype(np.float32)

        # Compute sum of squared values to be used in normalization
        if self.method == "user":
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=1)).ravel()
            end_local = self.n_rows
        elif self.method == "item":
            sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()
            end_local = self.n_columns
        else:
            raise NotImplementedError("Make sure 'method' in ['user', 'item']!")
        sumOfSquared = np.sqrt(sumOfSquared)

        start_block = 0

        # Compute all similarities using vectorization
        while start_block < end_local:
            end_block = min(start_block + block_size, end_local)
            this_block_size = end_block - start_block

            # All data points for a given user or item
            if self.method == "user":
                data = self.dataMatrix[start_block:end_block, :]
            else:
                data = self.dataMatrix[:, start_block:end_block]
            data = data.toarray()

            # Compute similarities

            if self.method == "user":
                this_block_weights = self.dataMatrix.dot(data.T)
            else:
                this_block_weights = self.dataMatrix.T.dot(data)

            for index_in_block in range(this_block_size):
                this_line_weights = this_block_weights[:, index_in_block]

                Index = index_in_block + start_block
                this_line_weights[Index] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:
                    denominator = (
                        sumOfSquared[Index] * sumOfSquared + self.shrink + 1e-6
                    )
                    this_line_weights = np.multiply(this_line_weights, 1 / denominator)

                elif self.shrink != 0:
                    this_line_weights = this_line_weights / self.shrink

                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of users or items
                # - Partition the data to extract the set of relevant users or items
                # - Sort only the relevant users or items
                # - Get the original index
                relevant_partition = (-this_line_weights).argpartition(self.TopK - 1)[
                    0 : self.TopK
                ]
                relevant_partition_sorting = np.argsort(
                    -this_line_weights[relevant_partition]
                )
                top_k_idx = relevant_partition[relevant_partition_sorting]
                neigh.append(top_k_idx)

                # Incrementally build sparse matrix, do not add zeros
                notZerosMask = this_line_weights[top_k_idx] != 0.0
                numNotZeros = np.sum(notZerosMask)

                values.extend(this_line_weights[top_k_idx][notZerosMask])
                if self.method == "user":
                    rows.extend(np.ones(numNotZeros) * Index)
                    cols.extend(top_k_idx[notZerosMask])
                else:
                    rows.extend(top_k_idx[notZerosMask])
                    cols.extend(np.ones(numNotZeros) * Index)

            start_block += block_size

        # End while
        if self.method == "user":
            W_sparse = sp.csr_matrix(
                (values, (rows, cols)),
                shape=(self.n_rows, self.n_rows),
                dtype=np.float32,
            )
        else:
            W_sparse = sp.csr_matrix(
                (values, (rows, cols)),
                shape=(self.n_columns, self.n_columns),
                dtype=np.float32,
            )
        return neigh, W_sparse.tocsc()

class ItemKNN(GeneralRecommender):
    r"""ItemKNN is a basic model that compute item similarity with the interaction matrix.
    Adjusting the value of 'knn_method' in the config file sets the method to either ItemKNN or UserKNN, respectively.
    """

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    # BINARY VERSION
    # def __init__(self, config, dataset):
    #     super(ItemKNN, self).__init__(config, dataset)
        
    #     # load parameters info
    #     self.k = config["k"]
    #     self.method = "item"
    #     self.shrink = config["shrink"] if "shrink" in config else 0.0
        
    #     # Load interaction matrix
    #     self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        
    #     # BINARIZE: Convert any non-zero value to 1
    #     self.interaction_matrix.data = np.ones_like(self.interaction_matrix.data)
        
    #     # Optional: Verify binarization
    #     print(f"Unique values after binarization: {np.unique(self.interaction_matrix.data)}")
        
    #     shape = self.interaction_matrix.shape
    #     assert self.n_users == shape[0] and self.n_items == shape[1]
        
    #     _, self.w = ComputeSimilarity(
    #         self.interaction_matrix, topk=self.k, shrink=self.shrink, method=self.method
    #     ).compute_similarity()
        
    #     if self.method == "user":
    #         self.pred_mat = self.w.dot(self.interaction_matrix).tolil()
    #     else:
    #         self.pred_mat = self.interaction_matrix.dot(self.w).tocsr()
        
    #     self.fake_loss = torch.nn.Parameter(torch.zeros(1))
    #     self.other_parameter_name = ["w", "pred_mat"]

    def __init__(self, config, dataset):
        super(ItemKNN, self).__init__(config, dataset)

        # load parameters info
        self.k = config["k"]
        # self.method = config["knn_method"]
        self.method = "item"
        self.shrink = config["shrink"] if "shrink" in config else 0.0
        self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)

        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        _, self.w = ComputeSimilarity(
            self.interaction_matrix, topk=self.k, shrink=self.shrink, method=self.method
        ).compute_similarity()

        if self.method == "user":
            self.pred_mat = self.w.dot(self.interaction_matrix).tolil()
        else:
            # self.pred_mat = self.interaction_matrix.dot(self.w).tolil()
            self.pred_mat = self.interaction_matrix.dot(self.w).tocsr()

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
            
            # Properly extract scalar from sparse matrix
            if isinstance(score, np.ndarray):
                score = score.item() if score.size == 1 else score[0]
            elif hasattr(score, 'toarray'):
                score = score.toarray()[0, 0]
            
            result.append(float(score))  # Ensure it's a Python float
        
        result = torch.tensor(result, dtype=torch.float32).to(self.device)
        return result

    # def predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     item = interaction[self.ITEM_ID]
    #     user = user.cpu().numpy().astype(int)
    #     item = item.cpu().numpy().astype(int)
    #     result = []

    #     for index in range(len(user)):
    #         uid = user[index]
    #         iid = item[index]
    #         score = self.pred_mat[uid, iid]
    #         result.append(score)
    #     result = torch.from_numpy(np.array(result)).to(self.device)
    #     return result

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user = user.cpu().numpy()
        score = self.pred_mat[user, :].toarray().flatten()
        result = torch.from_numpy(score).to(self.device)

        return result
    
class UserKNN(GeneralRecommender):
    
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(UserKNN, self).__init__(config, dataset)

        # load parameters info
        self.k = config["k"]
        self.method = "user"  # This is the key difference from ItemKNN
        self.shrink = config["shrink"] if "shrink" in config else 0.0

        self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        
        # Compute user-user similarity matrix
        _, self.w = ComputeSimilarity(
            self.interaction_matrix, topk=self.k, shrink=self.shrink, method=self.method
        ).compute_similarity()

        # For UserKNN: W is [n_users x n_users], so we do W.dot(interaction_matrix)
        # Result: [n_users x n_items]
        self.pred_mat = self.w.dot(self.interaction_matrix).tocsr()

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
            
            # Properly extract scalar from sparse matrix
            if isinstance(score, np.ndarray):
                score = score.item() if score.size == 1 else score[0]
            elif hasattr(score, 'toarray'):
                score = score.toarray()[0, 0]
            
            result.append(float(score))  # Ensure it's a Python float
        
        result = torch.tensor(result, dtype=torch.float32).to(self.device)
        return result

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user = user.cpu().numpy()

        score = self.pred_mat[user, :].toarray().flatten()
        result = torch.from_numpy(score).to(self.device)

        return result

class LightGCN_IPS(LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.ips_gamma = config["ips_gamma"]
        self.ips_min = config["ips_min"]
        self.ips_max_weight = config["ips_max_weight"]

        inter = dataset.inter_matrix(form="coo")

        counts = torch.bincount(
            torch.tensor(inter.col, dtype=torch.long),
            minlength=self.n_items
        ).float()

        # item 0 is normally RecBole's padding item
        counts = counts.clamp_min(1.0)

        propensity = counts / counts.max()
        propensity = propensity.pow(self.ips_gamma)
        propensity = propensity.clamp_min(self.ips_min)

        self.register_buffer("item_propensity", propensity)

    def calculate_loss(self, interaction):
        self.restore_user_e = None
        self.restore_item_e = None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(
            u_embeddings, pos_embeddings
        ).sum(dim=1)

        neg_scores = torch.mul(
            u_embeddings, neg_embeddings
        ).sum(dim=1)

        # BPR loss per observation
        bpr = -torch.log(
            torch.sigmoid(pos_scores - neg_scores) + 1e-8
        )

        propensity = self.item_propensity[pos_item]

        ips_weight = 1.0 / propensity
        ips_weight = torch.clamp(
            ips_weight,
            max=self.ips_max_weight
        )

        mf_loss = torch.mean(ips_weight * bpr)

        # same embedding regularization concept as LightGCN
        u_ego = self.user_embedding(user)
        pos_ego = self.item_embedding(pos_item)
        neg_ego = self.item_embedding(neg_item)

        reg_loss = (
            u_ego.norm(2).pow(2)
            + pos_ego.norm(2).pow(2)
            + neg_ego.norm(2).pow(2)
        ) / (2.0 * user.shape[0])

        return mf_loss + self.reg_weight * reg_loss

class EASE(GeneralRecommender):
    r"""EASE (Embarrassingly Shallow Autoencoders for Sparse Data)
    
    A linear autoencoder model that learns item-item similarities with a closed-form solution.
    
    Reference:
        Harald Steck. "Embarrassingly Shallow Autoencoders for Sparse Data." in WWW 2019.
    """

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(EASE, self).__init__(config, dataset)

        # Load parameters
        self.reg_weight = config["reg_weight"] if "reg_weight" in config else 500.0
        
        # Load interaction matrix (users x items)
        self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        
        # Binarize the interaction matrix
        self.interaction_matrix.data = np.ones_like(self.interaction_matrix.data)
        
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        
        print(f"Training EASE with {self.n_users} users and {self.n_items} items")
        print(f"Regularization weight: {self.reg_weight}")
        
        # Compute EASE item-item similarity matrix
        self.item_similarity = self._compute_ease_weights()
        
        # Precompute prediction matrix (users x items)
        print("Computing prediction matrix...")
        self.pred_mat = self.interaction_matrix.dot(self.item_similarity).tocsr()
        print("Prediction matrix computed")
        
        # Dummy parameters for RecBole compatibility
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["item_similarity", "pred_mat"]

    def _compute_ease_weights(self):
        """
        Compute EASE item-item similarity weights using closed-form solution.
        
        The EASE model solves:
        B = argmin ||X - XB||^2 + λ||B||^2
        subject to diag(B) = 0
        
        Where X is the user-item interaction matrix.
        """
        print("Computing EASE weights...")
        
        # Convert to dense for computation (only practical for smaller datasets)
        # For large datasets, you might need a sparse implementation
        X = self.interaction_matrix.toarray()
        
        # Compute Gram matrix: G = X^T X
        G = X.T.dot(X)
        
        # Add regularization to diagonal
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self.reg_weight
        
        # Solve for P: P = (X^T X + λI)^(-1)
        print("Inverting Gram matrix...")
        try:
            P = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            print("Matrix inversion failed, using pseudo-inverse...")
            P = np.linalg.pinv(G)
        
        # Compute B from P
        B = P / (-np.diag(P))
        
        # Set diagonal to zero (as per EASE constraint)
        B[diag_indices] = 0.0
        
        print("EASE weights computed")
        
        # Convert to sparse matrix for efficiency
        return sp.csr_matrix(B)

    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        """EASE has no training loss as it uses closed-form solution"""
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
            
            # Properly extract scalar from sparse matrix
            if isinstance(score, np.ndarray):
                score = score.item() if score.size == 1 else score[0]
            elif hasattr(score, 'toarray'):
                score = score.toarray()[0, 0]
            else:
                score = float(score)
            
            result.append(float(score))
        
        result = torch.tensor(result, dtype=torch.float32).to(self.device)
        return result

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user = user.cpu().numpy()
        
        # Get scores for all items for the given users
        score = self.pred_mat[user, :].toarray().flatten()
        result = torch.from_numpy(score).to(self.device)
        
        return result

class AdaptiveDiversity(GeneralRecommender):
    '''
        Backbone:
        - BPR
        - LightGCN

    '''

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.backbone_name = config["backbone"]

        # Create backbone recommender
        if self.backbone_name == "BPR":
            self.backbone = BPR(config, dataset)

        elif self.backbone_name == "LightGCN":
            self.backbone = LightGCN(config, dataset)

        else:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. "
                f"Supported: BPR, LightGCN"
            )

        # Build empirical distributions from current
        # RecBole training dataset

        self._build_diversity_statistics(dataset)

    @torch.no_grad()
    def update_diversity_state(self, user_id, item_id):
        """
        Update diversity statistics after ONE real simulated interaction.

        Important: Updates the empirical distributions used by AdaptiveDiversity model
        """

        if torch.is_tensor(user_id):
            user_id = int(user_id.item())

        if torch.is_tensor(item_id):
            item_id = int(item_id.item())

        if item_id == 0:
            raise ValueError(
                "Cannot update diversity state with RecBole padding item 0."
            )

        # Update counts
        self.interaction_counts[user_id, item_id] += 1.0
        self.user_totals[user_id, 0] += 1.0

        self.global_counts[item_id] += 1.0
        self.global_total += 1.0

        # Recompute q_u only for the affected user
        self.user_prob[user_id] = (
            self.interaction_counts[user_id]
            / self.user_totals[user_id, 0].clamp_min(1.0)
        )

        # Recompute q_U
        self.global_prob.copy_(
            self.global_counts
            / self.global_total.clamp_min(1.0)
        )

        eps = 1e-12

        # Recompute H(q_u) and tau_u
        p_user = self.user_prob[user_id]

        user_entropy = -torch.sum(
            torch.where(
                p_user > 0,
                p_user * torch.log(p_user + eps),
                torch.zeros_like(p_user)
            )
        )

        self.user_entropy[user_id] = user_entropy
        self.user_threshold[user_id] = torch.exp(
            -user_entropy
        )

        # Recompute H(q_U) and tau_U
        p_global = self.global_prob

        global_entropy = -torch.sum(
            torch.where(
                p_global > 0,
                p_global * torch.log(p_global + eps),
                torch.zeros_like(p_global)
            )
        )

        self.global_entropy.copy_(global_entropy)

        self.global_threshold.copy_(
            torch.exp(-global_entropy)
        )

    def _build_diversity_statistics(self, dataset):
        # Sparse user-item interaction matrix
        inter_matrix = dataset.inter_matrix(form="coo")

        users = torch.tensor(
            inter_matrix.row,
            dtype=torch.long
        )

        items = torch.tensor(
            inter_matrix.col,
            dtype=torch.long
        )

        # repeated interactions are explicitly preserved
        counts = torch.zeros(
            (self.n_users, self.n_items),
            dtype=torch.float32
        )

        ones = torch.ones(
            len(users),
            dtype=torch.float32
        )

        counts.index_put_(
            (users, items),
            ones,
            accumulate=True
        )

        # Individual distributions q_u
        user_totals = counts.sum(
            dim=1,
            keepdim=True
        )

        safe_user_totals = user_totals.clamp_min(1.0)
        user_prob = counts / safe_user_totals

        global_counts = counts.sum(dim=0)
        global_total = global_counts.sum()

        safe_global_total = global_total.clamp_min(1.0)
        global_prob = global_counts / safe_global_total

        self.register_buffer(
            "user_totals",
            user_totals
        )

        self.register_buffer(
            "global_total",
            global_total
        )

        # Save as buffers so they move with .to(device)
        self.register_buffer(
            "interaction_counts",
            counts
        )
        self.register_buffer(
            "user_prob",
            user_prob
        )
        self.register_buffer(
            "global_counts",
            global_counts
        )
        self.register_buffer(
            "global_prob",
            global_prob
        )

        # Compute thresholds once for this dataset state
        self._compute_entropy_thresholds()

    def _compute_entropy_thresholds(self):

        eps = 1e-12
        # Individual entropy H(q_u)
        user_p = self.user_prob
        user_entropy = -torch.sum(
            torch.where(
                user_p > 0,
                user_p * torch.log(user_p + eps),
                torch.zeros_like(user_p)
            ),
            dim=1
        )
        user_threshold = torch.exp(
            -user_entropy
        )

        # Collective entropy H(q_U)
        global_p = self.global_prob
        global_entropy = -torch.sum(
            torch.where(
                global_p > 0,
                global_p * torch.log(global_p + eps),
                torch.zeros_like(global_p)
            )
        )
        global_threshold = torch.exp(
            -global_entropy
        )
        self.register_buffer(
            "user_entropy",
            user_entropy
        )
        self.register_buffer(
            "user_threshold",
            user_threshold
        )
        self.register_buffer(
            "global_entropy",
            global_entropy
        )
        self.register_buffer(
            "global_threshold",
            global_threshold
        )

    def _get_admissible_mask(self, user):

        q_user = self.user_prob[user]

        tau_user = (
            self.user_threshold[user]
            .unsqueeze(1)
        )

        individual_mask = (
            q_user <= tau_user
        )

        collective_mask = (
            self.global_prob.unsqueeze(0)
            <= self.global_threshold
        ).expand(
            user.shape[0],
            -1
        ).clone()

        admissible_mask = (
            individual_mask
            & collective_mask
        )

        # RecBole padding
        individual_mask[:, 0] = False
        collective_mask[:, 0] = False
        admissible_mask[:, 0] = False

        return (
            individual_mask,
            collective_mask,
            admissible_mask
        )

    def calculate_loss(self, interaction):
        return self.backbone.calculate_loss(
            interaction
        )

    def predict(self, interaction):
        return self.backbone.predict(
            interaction
        )

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        
        # Base recommender relevance scores
        scores = self.backbone.full_sort_predict(
            interaction
        )
        scores = scores.view(
            user.shape[0],
            self.n_items
        )

        # Diversity admissibility
        (
            individual_mask,
            collective_mask,
            admissible_mask
        ) = self._get_admissible_mask(user)
        final_scores = scores.clone()

        for row in range(user.shape[0]):

            # Case 1:
            # A_u ∩ A_U is non-empty

            if admissible_mask[row].any():
                mask = admissible_mask[row]

            # Case 2:
            # intersection empty: prioritize collective diversity
            elif collective_mask[row].any():

                mask = collective_mask[row].clone()
                mask[0] = False

            # Case 3:
            # theoretical degenerate fallback: globally rarest item

            else:

                mask = torch.zeros(
                    self.n_items,
                    dtype=torch.bool,
                    device=scores.device
                )

                counts = self.global_counts.clone()

                # Never recommend RecBole padding
                counts[0] = float("inf")
                rarest_item = torch.argmin(
                    counts
                )
                mask[rarest_item] = True

            # Remove every inadmissible score
            final_scores[row][~mask] = -torch.inf

        # DEBUG
        # for row in range(user.shape[0]):
        #     user_id = int(user[row].item())
        #     print(
        #         f"[AdaptiveDiversity user={user_id}] "
        #         f"individual={individual_mask[row].sum().item()}, "
        #         f"collective={collective_mask[row].sum().item()}, "
        #         f"intersection={admissible_mask[row].sum().item()}"
        #     )
        #     print(
        #         f"tau_user={self.user_threshold[user_id].item():.6f}, "
        #         f"tau_global={self.global_threshold.item():.6f}"
        #     )

        #     n_items = self.n_items - 1  # exclude padding

        #     print(
        #         f"individual={individual_mask[row].sum().item()} "
        #         f"({100 * individual_mask[row].sum().item() / n_items:.2f}%)"
        #     )

        #     print(
        #         f"collective={collective_mask[row].sum().item()} "
        #         f"({100 * collective_mask[row].sum().item() / n_items:.2f}%)"
        #     )

        #     print(
        #         f"intersection={admissible_mask[row].sum().item()} "
        #         f"({100 * admissible_mask[row].sum().item() / n_items:.2f}%)"
        #     )

        #     h_user = self.user_entropy[user_id].item()
        #     h_global = self.global_entropy.item()

        #     print(
        #         f"H_user={h_user:.4f}, "
        #         f"effective_user_items={math.exp(h_user):.2f}"
        #     )

        #     print(
        #         f"H_global={h_global:.4f}, "
        #         f"effective_global_items={math.exp(h_global):.2f}"
        #     )

        #     only_individual = (
        #         individual_mask[row]
        #         & ~collective_mask[row]
        #     ).sum().item()

        #     only_collective = (
        #         collective_mask[row]
        #         & ~individual_mask[row]
        #     ).sum().item()

        #     neither = (
        #         ~individual_mask[row]
        #         & ~collective_mask[row]
        #     ).sum().item()

        #     print(
        #         f"individual_only={only_individual}, "
        #         f"collective_only={only_collective}, "
        #         f"neither={neither}"
        #     )

        #     danger_zone = (
        #         individual_mask[row]
        #         & ~collective_mask[row]
        #     )

        #     print(
        #         f"individual-but-not-collective="
        #         f"{danger_zone.sum().item()}"
        #     )

        #     global_p = self.global_prob.clone()

        #     valid = torch.arange(
        #         self.n_items,
        #         device=global_p.device
        #     ) != 0

        #     above = torch.where(
        #         valid & (global_p > self.global_threshold)
        #     )[0]

        #     below = torch.where(
        #         valid & (global_p <= self.global_threshold)
        #     )[0]

        #     print(
        #         f"globally above threshold: {len(above)}"
        #     )

        #     top_vals, top_idx = torch.topk(
        #         self.global_prob[1:],
        #         k=min(10, self.n_items - 1)
        #     )

        #     top_idx = top_idx + 1

        #     for iid, prob in zip(
        #         top_idx.tolist(),
        #         top_vals.tolist()
        #     ):
        #         print(
        #             f"item={iid}, "
        #             f"q_global={prob:.6f}, "
        #             f"tau_global={self.global_threshold.item():.6f}, "
        #             f"admissible={prob <= self.global_threshold.item()}"
        #         )

        #     selected_item = torch.argmax(
        #         final_scores[row]
        #     ).item()

        #     print(
        #         f"selected={selected_item}"
        #     )

        #     print(
        #         f"q_user(selected)="
        #         f"{self.user_prob[user_id, selected_item].item():.6f}"
        #     )

        #     print(
        #         f"tau_user="
        #         f"{self.user_threshold[user_id].item():.6f}"
        #     )

        #     print(
        #         f"q_global(selected)="
        #         f"{self.global_prob[selected_item].item():.6f}"
        #     )

        #     print(
        #         f"tau_global="
        #         f"{self.global_threshold.item():.6f}"
        #     )
        # exit()
        return final_scores.view(-1)