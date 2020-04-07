#Needs modifications

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__author__ = "Rajkumar Pujari"

class Encoder(nn.Module):
    
    def __init__(self, args):

        super(Encoder, self).__init__()

        self.args = args
        self.seq_rnn1 = nn.LSTM(input_size = args.d_encoder,
                               hidden_size = args.d_hidden,
                               num_layers = 1,
                               batch_first = True,
                               bias = False,
                               bidirectional = True) #Bi-LSTM1

        self.seq_rnn2 = nn.LSTM(input_size = 2 * args.d_hidden,
                               hidden_size = int(args.d_encoder / 2),
                               num_layers = 1,
                               batch_first = True,
                               bias = False,
                               bidirectional = True) #Bi-LSTM2
        
        if args.pretrained:
            #Load parmeters from path
            pass
           
    def forward(self, documents, doc_mask=None):
        '''
            Input:
                documents: batch * n_docs * n_sents * d_encoder
                doc_mask: batch * n_docs * n_sents
            Output:
                node_embs: batch * d_encoder
                
        '''
        bsize = documents.size(0)
        n_docs = documents.size(1)
        n_sents = documents.size(2)

        s_embs = documents.view(bsize, n_docs * n_sents, -1).contiguous()

        d_embs, _ = self.seq_rnn1(s_embs)
        d_embs = d_embs.view(bsize, n_docs, n_sents, -1).contiguous()
        if doc_mask is not None:
            s_mask = doc_mask
            s_mask = s_mask.unsqueeze(-1)
            s_mask = s_mask.expand(d_embs.size())
            d_embs = torch.sum(d_embs * s_mask, dim=-2) / torch.sum(s_mask, dim=-2)
        else:
            d_embs = torch.mean(d_embs, dim=-2)

        n_embs, _ = self.seq_rnn2(d_embs)
        if doc_mask is not None:
            d_mask = torch.sum(doc_mask, dim=-1)
            d_mask = d_mask.unsqueeze(-1)
            d_mask = d_mask.expand(n_embs.size())
            node_embs = torch.sum(n_embs * d_mask, dim=-2) / torch.sum(d_mask, dim=-2)
        else:
            node_embs = torch.mean(n_embs, dim=-2)

        return node_embs
