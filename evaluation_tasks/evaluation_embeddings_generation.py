#!/usr/bin/env python
# coding: utf-8

# @author: Rajkumar Pujari


import torch
import pickle
import numpy as np
from datetime import datetime
import copy
import random
import math
from sklearn.metrics import confusion_matrix
import csv
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import matplotlib.pyplot as plt


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[3]:

import sys
sys.path.append('./')
from compositional_learner.encoder import Encoder
from compositional_learner.composer import Composer


# In[4]:


from query_handling import graph_generation
import utils


# In[5]:


fpath = './data/composite_learner_data/'


# In[6]:


lp_sd = torch.load(fpath + 'saved_parameters/link_predictor_phase2_b99.pt', map_location='cpu')


# In[7]:


encoder_state_dict = {}
for key in lp_sd:
    if key.startswith('encoder'):
        encoder_state_dict[key[8:]] = lp_sd[key]
composer_state_dict = {}
for key in lp_sd:
    if key.startswith('composer'):
        composer_state_dict[key[9:]] = lp_sd[key]


# In[8]:


lp_bl_sd = torch.load(fpath + 'saved_parameters/link_predictor_baseline_phase2_b99.pt', map_location='cpu')


# In[9]:


bl_encoder_state_dict = {}
for key in lp_bl_sd:
    if key.startswith('encoder'):
        bl_encoder_state_dict[key[8:]] = lp_bl_sd[key]


# In[10]:


cuda_device = 0
class Args:
    def __init__(self):
        self.cuda = True
        self.d_encoder = 768
        self.d_hidden = 384
        self.n_heads = 12
        self.d_k = 64
        self.d_v = 64
        self.n_layers = 2
        self.dropout = 0.1
        self.n_classes = 2
        self.lr = 0.01
        self.weight_decay = 0
        self.pretrained = None #Pretrained state_dicts also include fine-tuning layers that we don't need
        
args = Args()


# In[11]:


class CompositionalReaderBaseline(nn.Module):
    
    def __init__(self, args, encoder_state_dict):
        super(CompositionalReaderBaseline, self).__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.encoder.load_state_dict(encoder_state_dict)
        
    def forward(self, node_docs, node_mask):
        self.eval()
        with torch.no_grad():
            if self.args.cuda:
                node_docs = node_docs.cuda()
                node_mask = node_mask.cuda()
            graph_node_embs = self.encoder(node_docs, node_mask)
            if self.args.cuda:
                graph_node_embs = graph_node_embs.cpu().numpy()
                updated_node_embs = [graph_node_embs]
                summary_emb = np.mean(graph_node_embs, axis=1)
            return graph_node_embs, updated_node_embs, summary_emb


# In[12]:


class CompositionalReader(nn.Module):
    
    def __init__(self, args, encoder_state_dict, composer_state_dict):
        super(CompositionalReader, self).__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.encoder.load_state_dict(encoder_state_dict)
        self.composer = Composer(args)
        self.composer.load_state_dict(composer_state_dict)
        
    def forward(self, node_docs, node_mask, comb_amtx):
        self.eval()
        with torch.no_grad():
            if self.args.cuda:
                node_docs = node_docs.cuda()
                node_mask = node_mask.cuda()
                comb_amtx = comb_amtx.cuda()
            graph_node_embs = self.encoder(node_docs, node_mask)
            updated_node_embs, summary_emb = self.composer(graph_node_embs, comb_amtx)
            if self.args.cuda:
                graph_node_embs = graph_node_embs.cpu().numpy()
                updated_node_embs = [nemb.cpu().numpy() for nemb in updated_node_embs]
                summary_emb = summary_emb.cpu().numpy()
            return graph_node_embs, updated_node_embs, summary_emb


# In[13]:


comp_reader = CompositionalReader(args, encoder_state_dict, composer_state_dict)


# In[14]:


comp_reader_baseline = CompositionalReaderBaseline(args, bl_encoder_state_dict)


# In[15]:

pol_ents = set()
issues = set()
with open(fpath + 'data_examples/queries.pkl', 'rb') as infile:
    queries = pickle.load(infile)
    for query in queries:
        pol_ents.add(query[0][0])
        issues.add(query[1][0])
print(issues)


# In[50]:


def get_coord(edata, ent, issue, baseline=False):
    edata['news'] = ({issue: {}}, {issue: {}}, {issue: {}})
    egraph = graph_generation.generate_graph(edata, threshold=2)
    egtensors = utils.get_graph_tensors(edata, egraph)
    doc_tensors, node_dids, graph_amtx = egtensors
    n2_ID = None
    for node in egraph[0]:
        if node.startswith('issue-' + issue):
            n1_ID = egraph[0][node]
        if node.startswith('doc-statement') or node.startswith('doc-tweet') or node.startswith('doc-quote'):
            n2_ID = egraph[0][node]
    if n2_ID:
        (doc_tensors, node_dids, graph_amtx, _, _), (rem_dids, rem_ids) = utils.trim_example(egraph, doc_tensors, node_dids, graph_amtx, n1_ID, n2_ID, trim_auth=True, ret_rem_ids=True)
        eg_x = [(doc_tensors, node_dids, graph_amtx, 0, 1)]
        eg_y = [1]
        node_docs, node_mask, comb_amtx, _, _, _ = utils.batch_to_tensors(eg_x, eg_y)
        with torch.cuda.device(cuda_device):
            with torch.no_grad():
                if baseline:
                    comp_reader_baseline.cuda()
                    out = comp_reader_baseline.forward(node_docs, node_mask)
                else:
                    comp_reader.cuda()
                    out = comp_reader.forward(node_docs, node_mask, comb_amtx)
        gn_embs, upd_embs, sum_embs = out
        return gn_embs, upd_embs, sum_embs, egraph, rem_ids, doc_tensors, node_dids, graph_amtx, node_docs, node_mask
    else:
        return None, None, None, None, [], None, None, None, None, None


# In[51]:

ent_embs_bert = {}
ent_embs_bl = {}
ent_embs_encoder = {}
ent_embs_model = {}
for issue in issues:
    ent_embs_bert[issue] = {}
    ent_embs_bl[issue] = {}
    ent_embs_encoder[issue] = {}
    ent_embs_model[issue] = {}
t1 = datetime.now()
incl_tweets = True
incl_quotes = True
incl_press = True
for query_batch in range(152):
    random.seed(4056)
    qdb_path = fpath + 'data_examples/query_dicts-' + str(query_batch) + '.pkl'
    with open(qdb_path, 'rb') as infile:
        query_dicts, _ = pickle.load(infile)
    batch_queries = queries[query_batch * 24: (query_batch + 1) * 24]
    #for each query
    for i, query in enumerate(batch_queries):
        ent, issue = query[0][0], query[1][0]
        eq_dict = copy.deepcopy(query_dicts[i])
        #dropping documents for ablation studies
        for key in eq_dict:
            if key not in ['news', 'description']:
                if not incl_tweets:
                    eq_dict[key]['tweets'] = ({}, {}, {})
                if not incl_quotes:
                    eq_dict[key]['quotes'] = ({}, {}, {})
                if not incl_press:
                    eq_dict[key]['statements'] = ({}, {}, {})
        #get the output of baseline (BERT adaptation) model
        out = get_coord(eq_dict, ent, issue, baseline=True)
        gn_embs, upd_embs, sum_embs, egraph, rem_ids, doc_tensors, node_dids, graph_amtx, node_docs, node_mask = out
        if len(rem_ids) > 0:
            for node in egraph[0]:
                if node == 'issue-' + issue:
                    n2_id = list(rem_ids).index(egraph[0][node])
                    n2_emb_bl = upd_embs[-1][0, n2_id, :]
                    node_len = int(torch.sum(node_mask[0, n2_id, :]).type(torch.LongTensor))
                    n2_emb_bert = torch.mean(node_docs[0, n2_id, :node_len, :], dim=0).numpy()
                if node.startswith('author-'):
                    n1_id = list(rem_ids).index(egraph[0][node])
                    n1_emb_bl = upd_embs[-1][0, n1_id, :]
                    node_len = int(torch.sum(node_mask[0, n1_id, :]).type(torch.LongTensor))
                    n1_emb_bert = torch.mean(node_docs[0, n1_id, :node_len, :], dim=0).numpy()

            ent_embs_bert[issue][ent] = [n1_emb_bert, n2_emb_bert]
            ent_embs_bl[issue][ent] = [n1_emb_bl, n2_emb_bl]

        #get output from encoder-composer model
        out = get_coord(query_dicts[i], ent, issue, baseline=False)
        gn_embs, upd_embs, sum_embs, egraph, rem_ids, doc_tensors, node_dids, graph_amtx, node_docs, node_mask = out
        if len(rem_ids) > 0:
            for node in egraph[0]:
                if node == 'issue-' + issue:
                    n2_id = list(rem_ids).index(egraph[0][node])
                    n2_emb_encoder = gn_embs[0, n2_id, :]
                    n2_emb_model = upd_embs[-1][0, n2_id, :]
                if node.startswith('author-'):
                    n1_id = list(rem_ids).index(egraph[0][node])
                    n1_emb_encoder = gn_embs[0, n1_id, :]
                    n1_emb_model = upd_embs[-1][0, n1_id, :]

            ent_embs_encoder[issue][ent] = [n1_emb_encoder, n2_emb_encoder]
            ent_embs_model[issue][ent] = [n1_emb_model, n2_emb_model]
            
    t2 = datetime.now()
    print(query_batch, 'done - ', t2-t1)


# In[52]:

with open('./data/evaluation_data/entity_issue_bert.pkl', 'wb') as outfile:
    pickle.dump(ent_embs_bert, outfile)
with open('./data/evaluation_data/entity_issue_bl.pkl', 'wb') as outfile:
    pickle.dump(ent_embs_bl, outfile)
with open('./data/evaluation_data/entity_issue_encoder.pkl', 'wb') as outfile:
    pickle.dump(ent_embs_encoder, outfile)
with open('./data/evaluation_data/entity_issue_model.pkl', 'wb') as outfile:
    pickle.dump(ent_embs_model, outfile)
