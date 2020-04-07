import torch
import torch.nn as nn
import torch.nn.functional as F
from compositional_learner.transformer.Layers import EncoderLayer

__author__ = "Rajkumar Pujari"

class GraphAttentionNetwork(nn.Module):
    '''transformer based graph attention network'''
    def __init__(self, d_node=768, d_hidden=384, n_heads=12, d_k=64, d_v=64, n_layers=2, dropout=0.1):
        
        super(GraphAttentionNetwork, self).__init__()

        self.d_node = d_node
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.layers = [EncoderLayer(d_node, d_hidden, n_heads, d_k, d_v, dropout=dropout) for _ in range(n_layers)]

        for i, layer in enumerate(self.layers):
            self.add_module('layer_{}'.format(i), layer)

        self.dropout = nn.Dropout(dropout)


    def forward(self, nodes, adj):
        out_embs = []
        out_attns = []
        updated_nodes = nodes
        for i, layer in enumerate(self.layers):
            updated_nodes, self_attn = layer(updated_nodes, slf_attn_mask=adj)
            out_embs.append(updated_nodes)
            out_attns.append(self_attn)

        return out_embs, out_attns
