import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_sequence

__author__ = "Rajkumar Pujari"

def batch_docs(doc_ids, dlens, batch_sent_num=10000):
    batches = []
    i = 0
    while i < len(doc_ids):
        batch = []
        nbatch = 0
        while nbatch < batch_sent_num:
            batch.append(doc_ids[i])
            nbatch += dlens[i]
            i += 1
            if i == len(doc_ids):
                break
        batches.append(batch)
    return batches

class Encoder(nn.Module):
    
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.args = args
        self.seq_rnn1 = nn.LSTM(input_size = args.d_encoder,
                               hidden_size = args.d_hidden,
                               num_layers = 1,
                               bias = False,
                               bidirectional = True) #Bi-LSTM1

        self.seq_rnn2 = nn.LSTM(input_size = 2 * args.d_hidden,
                               hidden_size = int(args.d_encoder / 2),
                               num_layers = 1,
                               bias = False,
                               bidirectional = True) #Bi-LSTM2
        
        i = 0
        for param in self.seq_rnn1.parameters():
            self.register_parameter('param-{}'.format(i), param)
            i += 1
        for param in self.seq_rnn2.parameters():
            self.register_parameter('param-{}'.format(i), param)
            i += 1


    def forward(self, graphs):
        '''
            Input:
                graphs: list of tuples of doc embs, node_doc_ids list
            Output:
                graphs_tensor: num_graphs * num_nodes * d_encoder
                graphs_mask: num_graphs * num_nodes
                
        '''
        num_graphs = len(graphs)
        num_nodes = max([len(graph[1]) for graph in graphs])
        graphs_tensor = torch.FloatTensor(num_graphs, num_nodes, self.args.d_encoder)
        graphs_mask = torch.zeros(num_graphs, num_nodes)
        if self.args.cuda:
            graphs_tensor = graphs_tensor.cuda()
            
        #For each graph
        for i, graph in enumerate(graphs):
            #Get docs, pass through RNN1 and compute doc embeddings as mean of RNN output of sent embs
            graph_docs = graph[0]
            doc_lens = [doc.size(0) for doc in graph_docs]
            #Use batching to avoid CUDA memory error
            doc_ids = list(range(len(graph[0])))
            doc_batches = batch_docs(doc_ids, doc_lens)
            doc_embs = []
            #Using packed_sequence to avoid padding, padding inflates the size of the tensor too much
            for doc_batch in doc_batches:
                batch_graph_docs = [graph_docs[j] for j in doc_batch]
                batch_doc_lens = [doc_lens[j] for j in doc_batch]
                torch.cuda.empty_cache()
                doc_packed_seq = pack_sequence(batch_graph_docs, enforce_sorted=False)
                doc_out, _ = self.seq_rnn1(doc_packed_seq)
                s = 0
                for l in batch_doc_lens:
                    doc_embs.append(torch.mean(doc_out.data[s:s+l, :], dim=0).unsqueeze(0))
                    s += l
            
            #Get the doc embs for each node, pass through RNN2 and compute node embs as mean of RNN output of doc embs
            node_doc_ids = graph[1]
            node_doc_embs = []
            node_lens = []
            for node in node_doc_ids:
                if len(node) == 0:
                    rtensor = torch.rand(1, 2 * self.args.d_hidden).type(torch.FloatTensor)
                    if self.args.cuda:
                        rtensor = rtensor.cuda()
                    node_doc_embs.append(rtensor)
                    node_lens.append(1)
                else:
                    nembs = [doc_embs[didx] for didx in node]
                    nemb = torch.cat(nembs, dim=0)
                    node_doc_embs.append(nemb)
                    node_lens.append(len(node))
            #Use batching to avoid CUDA memory error
            node_ids = list(range(len(node_lens)))
            node_batches = batch_docs(node_ids, node_lens)
            node_embs = []
            #Using packed_sequence to avoid padding, padding inflates the size of the tensor too much
            for node_batch in node_batches:
                batch_doc_embs = [node_doc_embs[j] for j in node_batch]
                batch_node_lens = [node_lens[j] for j in node_batch]
                torch.cuda.empty_cache()
                node_packed_seq = pack_sequence(batch_doc_embs, enforce_sorted=False)
                node_out, _ = self.seq_rnn2(node_packed_seq)
                s = 0
                for l in batch_node_lens:
                    node_embs.append(torch.mean(node_out.data[s:s+l, :], dim=0).unsqueeze(0))
                    s += l
                
            nodes_tensor = torch.cat(node_embs, dim=0)
            graphs_tensor[i, :nodes_tensor.size(0), :] = nodes_tensor
            graphs_mask[i, :nodes_tensor.size(0)] = 1
              
        return graphs_tensor, graphs_mask
