#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
from datetime import datetime
import copy
import random
import math
from sklearn.metrics import confusion_matrix
import sys
import torch
import json


# In[2]:

import sys
sys.path.append('./')
import utils
from query_handling import data_splitting_utils
from query_handling import graph_generation


# In[3]:


def get_doc_text(wparse):
    ret_text = []
    for para in wparse['tokens']:
        for sent in para:
            stext = ' '.join([tok['originalText'] for tok in sent['tokens']])
            ret_text.append(stext)
    return ret_text


fpath = './data/composite_learner_data/'

with open(fpath + 'formatted_tweet_parses.pkl', 'rb') as infile:
    tweet_parses = pickle.load(infile)
with open(fpath + 'tweet_doc_tensors_poprem.pkl', 'rb') as infile:
    tweet_doc_tensors_poprem, tweet_popents = pickle.load(infile)
tweet_texts = []
for ent in tweet_parses:
    for topic in tweet_parses[ent]:
        for event in tweet_parses[ent][topic]:
            for tparse in tweet_parses[ent][topic][event]:
                tweet_texts.append((tparse['parse_id'], get_doc_text(tparse['parse'])))
tweet_texts = sorted(tweet_texts, key=lambda x:x[0])
tweet_texts = [t[1] for t in tweet_texts]
print('Tweet data loaded.')
    
with open(fpath + 'stat_doc_tensors_poprem.pkl', 'rb') as infile:
    stat_doc_tensors_poprem, stat_popents = pickle.load(infile)
with open(fpath + 'stat_ent_mentions.pkl', 'rb') as infile:
    stat_ppaths, _, _ = pickle.load(infile)
stat_texts = [get_doc_text(json.load(open(stat_ppath))) for stat_ppath in stat_ppaths]
print('Statement data loaded.')
    
with open(fpath + 'news_doc_tensors_poprem.pkl', 'rb') as infile:
    news_doc_tensors_poprem, news_popents = pickle.load(infile)
with open(fpath + 'news_ent_mentions.pkl', 'rb') as infile:
    news_ppaths, _, _ = pickle.load(infile)
news_texts = [get_doc_text(json.load(open(news_ppath))) for news_ppath in news_ppaths]
print('News data loaded.')


# In[7]:


def is_same(r1_text, r2_text):
    if len(r1_text) != len(r2_text):
        return False
    else:
        for s1, s2 in zip(r1_text, r2_text):
            if s1.lower().strip() != s2.lower().strip():
                return False
        return True
    
def get_index(doc_text, ref_texts):
    for i, rt in enumerate(ref_texts):
        if is_same(doc_text, rt):
            return i
    return -1

def get_popular_entity(doc_ments):
    doc_ents = {}
    for sent in doc_ments:
        for ment in sent:
            if ment not in doc_ents:
                doc_ents[ment] = 0
            doc_ents[ment] += 1
    if len(doc_ents) > 0:
        doc_ent_list = [(ment, doc_ents[ment]) for ment in doc_ents]
        sorted_de_list = sorted(doc_ent_list, key=lambda x:x[1], reverse=True)
        return sorted_de_list[0][0]
    else:
        return None


# In[8]:


def make_examples(data_dict, data_cis, G, sel_cands, trim_auth=False, max_num=50):
    data_x = []
    data_y = []
    G_n, G_i2n, G_a = G
    num_ex = 0
    
    #For each selected candidate node
    for cand in sel_cands:
        if num_ex < max_num:
            ct = utils.split_node_name(cand)
            #Get the popular entity and popular entity-removed document embedding
            if ct[1] == 'news':
                _, dtype, topic, event, dd_idx = ct
                event = int(event)
                dd_idx = int(dd_idx)
                cand_text = data_dict['news'][0][topic][event][dd_idx]
                cand_idx = get_index(cand_text, news_texts)
                cand_poprem = news_popents[cand_idx]
                cand_rem_emb = news_doc_tensors_poprem[cand_idx]
                
            else:
                _, dtype, a2, topic, event, dd_idx = ct
                event = int(event)
                dd_idx = int(dd_idx)
                cand_text = data_dict[a2][dtype + 's'][0][topic][event][dd_idx]
                if dtype == 'tweet':
                    cand_idx = get_index(cand_text, tweet_texts)
                    cand_poprem = tweet_popents[cand_idx]
                    cand_rem_emb = tweet_doc_tensors_poprem[cand_idx]
                else:
                    cand_idx = get_index(cand_text, stat_texts)
                    cand_poprem = stat_popents[cand_idx]
                    cand_rem_emb = stat_doc_tensors_poprem[cand_idx]
                
            if cand_idx > 0 and cand_poprem:
                mn_name = None
                if cand_poprem in utils.graph_generation.wiki_ann_dict:
                    mn_name = 'ref-' + utils.graph_generation.get_wiki_name(cand_poprem)
                if mn_name in G_n and G_n[cand] in G_a[G_n[mn_name]]:
                    #Remove the link between the popular referenced entity and document in the graph
                    G_a[G_n[mn_name]].remove(G_n[cand])
                    doc_tensors, node_dids, graph_amtx = utils.get_graph_tensors(data_dict, G)
                    cand_did = node_dids[G_n[cand]][0]
                    #Replace the document embedding with the popular entity-removed document embedding
                    doc_tensors[cand_did] = np.squeeze(cand_rem_emb, axis=0)
                    n1_ID = G_n[mn_name]
                    n2_ID = G_n[cand]
                    #Add the positive example
                    eg, eg_size = utils.trim_example(G, doc_tensors, node_dids, graph_amtx, n1_ID, n2_ID, trim_auth=trim_auth)
                    data_x.append(eg)
                    data_y.append(1)
                    #Find another referenced entity that was not mentioned in the original document for negative examples
                    other_refs = [ID for ID in G_i2n.keys() if G_i2n[ID].startswith('ref-') and (n2_ID not in G_a[ID]) and ID != n1_ID]
                    neg_ref = random.choice(other_refs)
                    eg_doc_tensors, eg_node_dids, eg_graph_amtx, _, eg_n2_ID = eg
                    neg_eg = (eg_doc_tensors, eg_node_dids, eg_graph_amtx, neg_ref, eg_n2_ID)
                    data_x.append(neg_eg)
                    data_y.append(0)
                    #Add the link from entity to the popular entity back in the graph
                    G_a[G_n[mn_name]].append(G_n[cand])
                    num_ex += 1
        else:
            break
    return data_x, data_y


# In[9]:


def get_examples(query_dict, qd_split, trim_auth=False, num_tr=25):
    t, d, e = qd_split
    tc = utils.count_docs(t)
    tr_x = []
    tr_y = []
    de_x = []
    de_y = []
    te_x = []
    te_y = []
    
    train_dict, train_cis = data_splitting_utils.get_documents(query_dict, [t])
    train_graph = graph_generation.generate_graph(train_dict)
    news_nodes, tweet_nodes, statement_nodes, quote_nodes, event_nodes, author_nodes =                                                            utils.get_node_lists(train_graph, train_cis, t)
    cand_nodes = tweet_nodes + statement_nodes + news_nodes
    
    num_samples = min(2 * num_tr, len(cand_nodes))
    sel_cands = random.sample(cand_nodes, num_samples)
    train_x, train_y = make_examples(train_dict, train_cis, train_graph, sel_cands, trim_auth=trim_auth, max_num=num_tr)
    
    dev_dict, dev_cis = data_splitting_utils.get_documents(query_dict, [t, d])
    dev_graph = graph_generation.generate_graph(dev_dict)
    news_nodes, tweet_nodes, statement_nodes, quote_nodes, event_nodes, author_nodes =                                                            utils.get_node_lists(dev_graph, dev_cis, d)
    cand_nodes = tweet_nodes + statement_nodes + news_nodes
    num_samples = min(2 * num_tr, len(cand_nodes))
    sel_cands = random.sample(cand_nodes, num_samples)
    dev_x, dev_y = make_examples(dev_dict, dev_cis, dev_graph, sel_cands, trim_auth=trim_auth, max_num=0.2*num_tr)
    
    test_dict, test_cis = data_splitting_utils.get_documents(query_dict, [t, d, e])
    test_graph = graph_generation.generate_graph(test_dict)
    news_nodes, tweet_nodes, statement_nodes, quote_nodes, event_nodes, author_nodes =                                                            utils.get_node_lists(test_graph, test_cis, e)
    cand_nodes = tweet_nodes + statement_nodes + news_nodes
    num_samples = min(2 * num_tr, len(cand_nodes))
    sel_cands = random.sample(cand_nodes, num_samples)
    test_x, test_y = make_examples(test_dict, test_cis, test_graph, sel_cands, trim_auth=trim_auth, max_num=0.2*num_tr)
    
    
    return (train_x, train_y), (dev_x, dev_y), (test_x, test_y)


# In[11]:


def batchify_sents(all_sents, batch_size=100):
    batches = []
    beg = 0
    end = batch_size
    while beg < len(all_sents):
        batches.append(all_sents[beg:end])
        beg = end
        end += batch_size
    return batches


# In[12]:


def compute_embeddings(all_docs, batch_size=50, log=sys.stdout):
    ret_embs = []
    doc_lens = [len(doc) for doc in all_docs]
    all_sents = []
    for doc in all_docs:
        all_sents += doc
    batches = batchify_sents(all_sents, batch_size=batch_size) 
    
    print(len(batches), ' batches created.', file=log, flush=True)
    
    tot = 0
    bnum = 0
    t1 = datetime.now()
    all_sents_tensor = torch.FloatTensor(0, 768)
    
    for batch in batches:
        try:
            batch_tensor = create_bert_emb(batch)
        except:
            batch_tensor = torch.FloatTensor(len(batch), 768)
        all_sents_tensor = torch.cat([all_sents_tensor, batch_tensor], dim=0)
        bnum += 1
        tot += len(batch)
        if bnum % 500 == 0:
            t2 = datetime.now()
            print(tot, '/', len(all_sents), ' done - ', t2 - t1, file=log, flush=True)
    
    b = 0
    for dlen in doc_lens:
        e = b + dlen
        ret_embs.append(all_sents_tensor[b:e, :].data.numpy())
        b = e

    return ret_embs


# In[10]:


def load_data_batch(qdb_path, log=sys.stdout, trim_auth=False):
    with open(qdb_path, 'rb') as infile:
        query_dicts, qd_splits = pickle.load(infile)
    i = 0
    train_x = []
    train_y = []
    dev_x = []
    dev_y = []
    test_x = []
    test_y = []
    t1 = datetime.now()
    for query_dict, qd_split in zip(query_dicts, qd_splits):
        t, d, e = get_examples(query_dict, qd_split, trim_auth=True)
        train_x += t[0]
        train_y += t[1]
        dev_x += d[0]
        dev_y += d[1]
        test_x += e[0]
        test_y += e[1]
        i += 1
        t2 = datetime.now()
        print(i, 'done', t2-t1, file=log, flush=True)
    
    print(len(train_x), len(dev_x), len(test_x), file=log, flush=True)
   
    data = ((train_x, train_y), (dev_x, dev_y), (test_x, test_y))
    return data

