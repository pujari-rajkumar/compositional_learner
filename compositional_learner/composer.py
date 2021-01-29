import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append('./')
from compositional_learner.graph_attention_network import GraphAttentionNetwork


__author__ = "Rajkumar Pujari"

class Composer(nn.Module):
    "given nodes and adjacency matrix, composer generates contextualized node embeddings and summary embedding"
    def __init__(self, args):
        super(Composer, self).__init__()

        self.args = args
        self.composer = GraphAttentionNetwork(args.d_encoder, args.d_hidden, args.n_heads, args.d_k, args.d_v, args.n_layers, args.dropout)
           
    def forward(self, nodes, adj, get_attn=False):
        '''
            Input:
                nodes : batch * n_nodes * d_node
                adj : batch * n_nodes * n_nodes
                
            Output:
                ctx_node_embs: [batch * n_nodes * d_node] * n_layers
                summary_emb: batch * d_node
                node_attn: batch * nheads * n_nodes * n_nodes
        '''
        
        ctx_node_embs, node_attn = self.composer(nodes, adj)
        final_layer = ctx_node_embs[-1]
        summary_emb = torch.mean(final_layer, dim=1)

        if get_attn:
            return ctx_node_embs, summary_emb, node_attn

        return ctx_node_embs, summary_emb
