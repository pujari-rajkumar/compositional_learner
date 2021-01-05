import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

__author__ = "Rajkumar Pujari"

class Encoder(nn.Module):
    
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.args = args

        self.seq_rnn = nn.LSTM(input_size = args.d_encoder,
                               hidden_size = int(args.d_encoder / 2),
                               num_layers = 1,
                               bias = False,
                               batch_first=True,
                               bidirectional = True)


    def forward(self, node_docs, node_mask):
        '''
            Input:
                node_docs: batch_size * num_nodes * num_docs * doc_size
                node_mask: batch_size * num_nodes * num_docs
            Output:
                node_embs: batch_size * num_nodes * d_encoder
                
        '''
        batch_size = node_docs.size(0)
        num_nodes = node_docs.size(1)
        num_docs = node_docs.size(2)
        doc_size = node_docs.size(3)
        
        node_docs = node_docs.view(batch_size * num_nodes, num_docs, doc_size)
        node_out, _ = self.seq_rnn(node_docs)
        node_out = node_out.transpose(1, 2).contiguous()
        node_mask = node_mask.view(batch_size * num_nodes, num_docs, 1)
        node_embs = node_out.bmm(node_mask).view(batch_size * num_nodes, self.args.d_encoder)
        node_embs /= torch.sum(node_mask, dim=1)
        node_embs = node_embs.view(batch_size, num_nodes, self.args.d_encoder)
        
        return node_embs
