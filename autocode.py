import pandas as pd
import torch.nn as nn
import torch
import json
import numpy as np
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import (
    BaseFactorizationMachine,
    MLPLayers,
)
from recbole.quick_start.quick_start import load_data_and_model
from recbole.data.interaction import Interaction
from recbole.utils import InputType, FeatureType
import torch.nn.functional as F


class AutoCode(ContextRecommender):

    def __init__(self, config, dataset):
        super(AutoCode, self).__init__(config, dataset)

        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]

        self.row, self.col = torch.triu_indices(
            config["train_batch_size"], config["train_batch_size"], offset=1
        )

        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]

        self.num_code_book = config["num_code_book"]
        self.book_dim = config["book_size"]
        self.item_code_dims = sum(self.book_dim)
        self.token_code_embedding_table = nn.Embedding(
            self.item_code_dims, self.embedding_size // self.num_code_book
        )
        self.token_user_embedding_table = nn.Embedding(
            self.item_code_dims, self.embedding_size // self.num_code_book
        )
        self.first_order_linear_code = nn.Embedding(self.item_code_dims, 1)
        self.first_order_linear_user = nn.Embedding(self.item_code_dims, 1)

        self.cl_loss_weight = config["cl_loss_weight"]
        self.al_loss_weight = config["al_loss_weight"]

        self.max_seq_length = self.num_code_book

        self.user_feat = dataset.get_user_feature()
        self.item_feat = dataset.get_item_feature()

        self.num_user = dataset.num("user_id")
        self.num_item = dataset.num("item_id")

        self.attention_size = config["attention_size"]
        self.attn_dropout = config["attn_dropout"]
        self.n_layers = config["n_layers"]
        self.num_heads = config["num_heads"]
        self.has_residual = config["has_residual"]
        self.cl_loss_cate = config["cl_loss_cate"]

        self.atten_output_dim = 2 * self.attention_size
        self.att_embedding = nn.Linear(self.embedding_size, self.attention_size)

        self.self_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.attention_size, self.num_heads, dropout=self.attn_dropout
                )
                for _ in range(self.n_layers)
            ]
        )
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)

        self.att_ln = nn.LayerNorm(self.atten_output_dim)

        self.bn_output = nn.BatchNorm1d(1)
        self.epsilon = 1e-8

        dataset_name = config["dataset"]


        if dataset_name == "MovielensLatest_x1":
            item_code_path = "/data/oukeshan/RecBole/code_data/tag/all_item_codes_256x2_lgn.npy"
            user_code_path = "/data/oukeshan/RecBole/code_data/tag/all_user_codes_256x2_lgn.npy"
            item_embedding_init_path = "/data/oukeshan/RecBole/code_data/tag/item_code_embedding_init_256x2_lgn.npy"
            user_embedding_init_path = "/data/oukeshan/RecBole/code_data/tag/user_code_embedding_init_256x2_lgn.npy"

        elif dataset_name == "ml-1m":
            item_code_path = "/data/oukeshan/RecBole/code_data/movielens/all_item_codes_512x1_lgn.npy"
            user_code_path = "/data/oukeshan/RecBole/code_data/movielens/all_user_codes_512x1_lgn.npy"
            item_embedding_init_path = "/data/oukeshan/RecBole/code_data/movielens/item_code_embedding_init_512x1_lgn.npy"
            user_embedding_init_path = "/data/oukeshan/RecBole/code_data/movielens/user_code_embedding_init_512x1_lgn.npy"

        elif dataset_name == "Frappe":
            item_code_path = "/data/oukeshan/RecBole/code_data/frappe/all_item_codes_256x1_bpr.npy"
            user_code_path = "/data/oukeshan/RecBole/code_data/frappe/all_user_codes_256x1_bpr.npy"
            item_embedding_init_path = "/data/oukeshan/RecBole/code_data/frappe/item_code_embedding_init_256x1_bpr.npy"
            user_embedding_init_path = "/data/oukeshan/RecBole/code_data/frappe/user_code_embedding_init_256x1_bpr.npy"

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")


        self.item_code = torch.tensor(np.load(item_code_path)).to(self.device)
        self.user_code = torch.tensor(np.load(user_code_path)).to(self.device)

        self.item_embedding_init = torch.tensor(np.load(item_embedding_init_path)).to(
            self.device
        )
        self.user_embedding_init = torch.tensor(np.load(user_embedding_init_path)).to(
            self.device
        )

        self.num_feature_field += 2

        if self.has_residual:
            self.v_res_embedding = torch.nn.Linear(
                self.embedding_size, self.attention_size
            )

        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [
            self.embedding_size * self.num_feature_field
        ] + self.mlp_hidden_size  # [mlp 256,256,256]
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)

        self.deep_predict_layer = nn.Linear(
            self.mlp_hidden_size[-1], 1
        )  # Linear product to the final score
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        self.apply(self._init_weights)

        with torch.no_grad():
            self.token_code_embedding_table.weight.copy_(self.item_embedding_init)
            self.token_user_embedding_table.weight.copy_(self.user_embedding_init)

    def autoint_layer(self, infeature):
        """Get the attention-based feature interaction score

        Args:
            infeature (torch.FloatTensor): input feature embedding tensor. shape of[batch_size,field_size,embed_dim].

        Returns:
            torch.FloatTensor: Result of score. shape of [batch_size,1] .
        """

        att_infeature = self.att_embedding(infeature)
        cross_term = att_infeature.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)  # [batch_size,num_feature,att_dim]

        # Residual connection
        if self.has_residual:
            v_res = self.v_res_embedding(infeature)
            cross_term += v_res
        # Interacting layer
        cross_term = cross_term.contiguous().view(-1, self.atten_output_dim)
        cross_term = self.att_ln(cross_term)
        cross_term = F.relu(cross_term)

        att_output = self.attn_fc(cross_term)

        return att_output

    def _compute_indices(self, batch_size):
        row, col = [], []
        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                row.append(i)
                col.append(j)
        return torch.tensor(row), torch.tensor(col)

    def compute_alignment_loss(self, x_emb):
        alignment_loss = (
            torch.norm(x_emb[self.row].sub(x_emb[self.col]), dim=2).pow(2).mean()
        )
        return alignment_loss

    def compute_uniformity_loss(self, x_emb):
        frac = torch.matmul(x_emb, x_emb.transpose(2, 1))  # B,F,F
        denom = torch.matmul(
            torch.norm(x_emb, dim=2).unsqueeze(2), torch.norm(x_emb, dim=2).unsqueeze(1)
        )  # 64，30,30
        res = torch.div(frac, denom + 1e-4)
        uniformity_loss = res.mean()
        return uniformity_loss

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def compute_all_alignment_loss(self, embedds):
        """
        Calculate feature alignment loss based on all feature representation.
        """
        embedds = embedds
        field_dims = self.book_dim * 2
        field_dims_cum = np.array((0, *np.cumsum(field_dims)))
        alignment_loss = 0.0
        pairs = 0
        for i, (start, end) in enumerate(zip(field_dims_cum[:-1], field_dims_cum[1:])):
            embed_f = embedds[start:end, :]
            # print(embed_f)
            loss_f = 0.0
            for j in range(field_dims[i]):
                loss_f += torch.norm(embed_f[j, :].sub(embed_f), dim=1).pow(2).sum()
            pairs += field_dims[i] * field_dims[i]
            alignment_loss += loss_f

        alignment_loss /= pairs
        return alignment_loss

    def compute_all_uniformity_loss(self, embedds):
        """
        Calculate field uniformity loss based on all feature representation.
        """
        embedds = embedds
        field_dims = self.book_dim * 2
        field_dims_cum = np.array((0, *np.cumsum(field_dims)))
        field_len = embedds.size()[0]
        field_index = np.array(range(field_len))
        uniformity_loss = 0.0
        #     for i in
        pairs = 0
        for i, (start, end) in enumerate(zip(field_dims_cum[:-1], field_dims_cum[1:])):
            index_f = np.logical_and(field_index >= start, field_index < end)
            embed_f = embedds[index_f, :]
            embed_not_f = embedds[~index_f, :]
            frac = torch.matmul(embed_f, embed_not_f.transpose(1, 0))  # f1,f2
            denom = torch.matmul(
                torch.norm(embed_f, dim=1).unsqueeze(1),
                torch.norm(embed_not_f, dim=1).unsqueeze(0),
            )  # f1,f2
            res = torch.div(frac, denom + 1e-4)
            uniformity_loss += res.sum()
            pairs += (field_len - field_dims[i]) * field_dims[i]
        uniformity_loss /= pairs
        return uniformity_loss

    def forward(self, interaction):

        user_id = interaction["user_id"]
        item_id = interaction["item_id"]

        item_codes = self.item_code[item_id].to(self.device)

        token_emb_table = self.token_embedding_table.embedding.weight

        user_embedding = token_emb_table[: self.num_user]
        item_embedding = token_emb_table[self.num_user : self.num_user + self.num_item]

        user_emb = user_embedding[user_id].unsqueeze(1)
        item_emb = item_embedding[item_id].unsqueeze(1)

        user_codes = self.user_code[user_id].to(self.device)

        deepfm_all_embeddings = self.concat_embed_input_fields(interaction)

        deepfm_code_embeddings_cl = self.token_code_embedding_table(item_codes)
        deepfm_code_embeddings_cl = deepfm_code_embeddings_cl.view(
            deepfm_code_embeddings_cl.shape[0], -1
        ).unsqueeze(1)
        deepfm_code_embeddings = self.alpha * deepfm_code_embeddings_cl

        code_infeature = torch.cat([deepfm_code_embeddings, item_emb], dim=1)
        item_emb_temp = self.autoint_layer(code_infeature)  # [b,1] for prediction

        deepfm_user_embeddings_cl = self.token_user_embedding_table(user_codes)
        deepfm_user_embeddings_cl = deepfm_user_embeddings_cl.view(
            deepfm_user_embeddings_cl.shape[0], -1
        ).unsqueeze(1)
        deepfm_user_embeddings = self.beta * deepfm_user_embeddings_cl

        user_infeature = torch.cat([deepfm_user_embeddings, user_emb], dim=1)

        user_emb_temp = self.autoint_layer(user_infeature)  # [b,1] for prediction

        deepfm_concat_embeddings = torch.cat(
            [deepfm_all_embeddings, deepfm_code_embeddings, deepfm_user_embeddings],
            dim=1,
        )
        deepfm_cl_embeddings = torch.cat(
            [deepfm_code_embeddings_cl, deepfm_user_embeddings_cl], dim=1
        )

        batch_size = deepfm_concat_embeddings.shape[0]

        if self.cl_loss_cate == "align_loss": 
            if batch_size != 4096:
                self.row, self.col = torch.triu_indices(batch_size, batch_size, offset=1)

            cl_loss = self.compute_alignment_loss(deepfm_cl_embeddings)

        elif self.cl_loss_cate == "unifor_loss":
            cl_loss = self.compute_uniformity_loss(deepfm_cl_embeddings)

        code_linear = self.first_order_linear_code(item_codes)
        code_linear = self.alpha * torch.sum(code_linear, dim=1, keepdim=True).squeeze(
            dim=-1
        )

        user_linear = self.first_order_linear_user(user_codes)
        user_linear = self.beta * torch.sum(user_linear, dim=1, keepdim=True).squeeze(
            dim=-1
        )

        y_fm = (
            self.first_order_linear(interaction)
            + code_linear
            + user_linear
            + self.fm(deepfm_concat_embeddings)
        )

        y_deep = self.deep_predict_layer(
            self.mlp_layers(deepfm_concat_embeddings.view(batch_size, -1))
        )

        y = y_deep + y_fm + self.gamma * (item_emb_temp + user_emb_temp)

        y_code = item_emb_temp + user_emb_temp
        y_origin = y_deep + y_fm



        return_dict = {
            "y": y.squeeze(-1),
            "y_code": y_code.squeeze(-1),
            "y_origin": y_origin.squeeze(-1),
            "cl_loss": cl_loss,
        }

        return return_dict

    def calculate_loss(self, interaction):

        label = interaction[self.LABEL]

        return_dict = self.forward(interaction)

        output = return_dict["y"]
        cl_loss = return_dict["cl_loss"]
        y_code = return_dict["y_code"]
        y_origin = return_dict["y_origin"]

        code_loss = self.loss(y_code, label)
        origin_loss = self.loss(y_origin, label)

        ctr_loss = self.loss(output, label)
        loss = (
            ctr_loss
            + cl_loss * self.cl_loss_weight
            + self.al_loss_weight * (code_loss + origin_loss)
        )

        return loss

    def predict(self, interaction):
        return_dict = self.forward(interaction)
        prediction = self.sigmoid(return_dict["y"])

        return prediction