#!/usr/bin/env python
# coding: utf-8

# @author: Rajkumar Pujari


import pickle
import os
import json
import threading
import math
import torch
from transformers import *
from datetime import datetime
import time
import re
from nltk.corpus import stopwords


cuda_device = 0

def get_popular_entity(doc_sref, eset):
    doc_ents = {}
    max_freq = -1
    for para in doc_sref:
        for sent in doc_sref[para]:
            for ent in doc_sref[para][sent]:
                if ent not in doc_ents:
                    doc_ents[ent] = 0
                doc_ents[ent] += 1
                max_freq = max(max_freq, doc_ents[ent])
    if len(doc_ents) > 0:
        sorted_dents = sorted([k for k in doc_ents.keys() if doc_ents[k] == max_freq], key=lambda x:eset[x][0], reverse=True)
        return sorted_dents[0]
    else:
        return None



def remove_entity(doc_text, ent):
    ret_text = []
    ent = ent.lower().strip()
    for sent in doc_text:
        stext = ' '.join(sent).lower().strip()
        stext = stext.replace(ent, '<ent>')
        ret_text.append(stext.split())
    return ret_text



def get_doc_text(pparse, is_path=True):
    if is_path:
        pparse = json.load(open(pparse, 'r'))
    ret_text = []
    for para in pparse['tokens']:
        for sent in para:
            stext = ' '.join([tok['originalText'] for tok in sent['tokens']])
            ret_text.append(stext)
    return ret_text



def concat_doc_tensors(d_tensors):
    s_len = max([t.size(1) for t in d_tensors])
    emb_size = d_tensors[0].size(-1)
    ret_tensor = torch.FloatTensor(len(d_tensors), s_len, emb_size)
    ret_mask = torch.zeros(len(d_tensors), s_len)
    for i, d_tensor in enumerate(d_tensors):
        ret_tensor[i, :d_tensor.size(1), :] = d_tensor
        ret_mask[i, :d_tensor.size(1)] = 1
    return ret_tensor, ret_mask



def create_bert_emb(all_sents, tok_pooling='mean', get_cls_emb=False):
    if len(all_sents) > 0:
        with torch.cuda.device(0):
            all_toks = emb_tokenizer.batch_encode_plus(all_sents, padding='longest',\
                                                   add_special_tokens=True)
            tok_tensor = torch.tensor(all_toks['input_ids']).to('cuda')
            tok_tensor = tok_tensor[:, :512]
            with torch.no_grad():
                model_out = emb_model(tok_tensor)
                all_doc_tensor = model_out[0]
                if get_cls_emb:
                    all_doc_tensor = model_out[1]
                all_doc_tensor = all_doc_tensor.to('cpu')
            if get_cls_emb:
                return all_doc_tensor
            all_attn_mask = torch.tensor(all_toks['attention_mask'])
            ret_tensor = torch.FloatTensor(all_doc_tensor.size(0), all_doc_tensor.size(-1))
            for i in range(all_doc_tensor.size(0)):
                slen = torch.sum(all_attn_mask[i, :])
                if tok_pooling == 'mean':
                    ret_tensor[i, :] = torch.mean(all_doc_tensor[i, :slen, :], dim=0)
                elif tok_pooling == 'sum':
                    ret_tensor[i, :] = torch.sum(all_doc_tensor[i, :slen, :], dim=0)
                else:
                    return 'invalid tok pooling'
            return ret_tensor


# In[ ]:


def batchify(all_sents, batch_size=100):
    batches = []
    beg = 0
    end = batch_size
    while beg < len(all_sents):
        batches.append(all_sents[beg:end])
        beg = end
        end += batch_size
    return batches



def get_bert_embs(ppaths, is_parses=True, batch_size=100, max_sent_len=150, popular_entities=None):
    ret_embs = []
    t1 = datetime.now()
    all_sents = []
    doc_lens = []
    for i, ppath in enumerate(ppaths):
        if not is_parses:
            doc_text = get_doc_text(ppath, is_path=True)
        else:
            doc_text = get_doc_text(ppath, is_path=False)
        doc_lens.append(len(doc_text))
        doc_text = [sent.split()[:max_sent_len] for sent in doc_text]
        if popular_entities and popular_entities[i]:
            doc_text = remove_entity(doc_text, popular_entities[i])
        all_sents += doc_text
        if (i + 1) % 2500 == 0:
            t2 = datetime.now()
            print(i, '/', len(ppaths), ' texts extracted - ', t2 - t1)
            
    batches = batchify(all_sents, batch_size=batch_size) 
    print(len(batches), ' batches created.')
    
    tot = 0
    bnum = 0
    t1 = datetime.now()
    all_sents_tensor = torch.FloatTensor(0, 768)
    
    for batch in batches:
        batch_tensor = create_bert_emb(batch)
        all_sents_tensor = torch.cat([all_sents_tensor, batch_tensor], dim=0)
        bnum += 1
        tot += len(batch)
        if bnum % 5 == 0:
            t2 = datetime.now()
            print(tot, '/', len(all_sents), ' done - ', t2 - t1)
    
    b = 0
    for dlen in doc_lens:
        e = b + dlen
        ret_embs.append(all_sents_tensor[b:e, :].unsqueeze(0).data.numpy())
        b = e

    return ret_embs



fpath = './data/composite_learner_data/'

tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
with torch.cuda.device(cuda_device):
    with torch.no_grad():
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=False, output_attentions=False)
        model.eval()
        model.to('cuda')



# # Tweet embedding computation

with open(fpath + 'formatted_tweet_parses.pkl', 'rb') as infile:
    formatted_tweet_parses = pickle.load(infile)
with open(fpath + 'formatted_tweet_ent_mentions.pkl', 'rb') as infile:
    _, ft_esets, ft_sref_dict = pickle.load(infile)


tweet_parse_list = []
for ent in formatted_tweet_parses:
    for topic in formatted_tweet_parses[ent]:
        for event in formatted_tweet_parses[ent][topic]:
            for tweet in formatted_tweet_parses[ent][topic][event]:
                tweet_parse_list.append((tweet['parse'], tweet['parse_id']))
sorted_tp_list = sorted(tweet_parse_list, key=lambda x:x[1])
sorted_tps = [x[0] for x in sorted_tp_list]

t1 = datetime.now()

if not os.path.exists(fpath + 'tweet_doc_tensors.pkl'):
    dembs = get_bert_embs(sorted_tps, is_parses=True, batch_size=100, max_sent_len=150)
    with open(fpath + 'tweet_doc_tensors.pkl', 'wb') as outfile:
        pickle.dump(dembs, outfile)

if not os.path.exists(fpath + 'tweet_doc_tensors_poprem.pkl'):
    tweet_popular_entities = [get_popular_entity(ft_sref_dict[i], ft_esets) for i in range(len(sorted_tps))]
    dembs = get_bert_embs(sorted_tps, is_parses=True, batch_size=100, max_sent_len=150, popular_entities=tweet_popular_entities)
    with open(fpath + 'tweet_doc_tensors_poprem.pkl', 'wb') as outfile:
        pickle.dump((dembs, tweet_popular_entities), outfile)

t2 = datetime.now()
print('Tweet embeddings computed. Time taken: ', t2 - t1)



# # Quote embedding computation

with open(fpath + 'formatted_quote_data.pkl', 'rb') as infile:
    formatted_quote_data = pickle.load(infile)
with open(fpath + 'quote_ent_mentions.pkl', 'rb') as infile:
    quote_ppaths, q_eset, q_sref_dict = pickle.load(infile)


quote_ppaths_rel = []
for ppath in quote_ppaths:
    ppath = ppath.replace('/homes/rpujari/scratch/', './data/')
    quote_ppaths_rel.append(ppath)

print(quote_ppaths_rel[0])
t1 = datetime.now()

if not os.path.exists(fpath + 'quote_doc_tensors.pkl'):
    dembs = get_bert_embs(quote_ppaths_rel, is_parses=False, batch_size=100, max_sent_len=150)
    with open(fpath + 'quote_doc_tensors.pkl', 'wb') as outfile:
        pickle.dump(dembs, outfile)

t2 = datetime.now()
print('Quote embeddings computed. Time taken: ', t2 - t1)



# # News embedding computation

with open(fpath + 'formatted_news_data.pkl', 'rb') as infile:
    formatted_news_data = pickle.load(infile)
with open(fpath + 'news_ent_mentions.pkl', 'rb') as infile:
    news_ppaths, n_eset, n_sref_dict = pickle.load(infile)

news_ppaths_rel = []
for ppath in news_ppaths:
    ppath = ppath.replace('/homes/rpujari/scratch/', './data/')
    news_ppaths_rel.append(ppath)

print(news_ppaths_rel[0])

t1 = datetime.now()

if not os.path.exists(fpath + 'news_doc_tensors.pkl'):
    dembs = get_bert_embs(news_ppaths_rel, is_parses=False, batch_size=100, max_sent_len=150)
    with open(fpath + 'news_doc_tensors.pkl', 'wb') as outfile:
        pickle.dump(dembs, outfile)

if not os.path.exists(fpath + 'news_doc_tensors_poprem.pkl'):
    news_popular_entities = [get_popular_entity(n_sref_dict[i], n_eset) for i in range(len(news_ppaths))]
    dembs = get_bert_embs(news_ppaths_rel, is_parses=False, batch_size=100, max_sent_len=150, popular_entities=news_popular_entities)
    with open(fpath + 'news_doc_tensors_poprem.pkl', 'wb') as outfile:
        pickle.dump((dembs, news_popular_entities), outfile)

t2 = datetime.now()
print('News embeddings computed. Time taken: ', t2 - t1)



# # Wiki embedding computation

with open(fpath + 'wiki_ent_mentions.pkl', 'rb') as infile:
    wiki_ppaths, w_eset, w_sref_dict = pickle.load(infile)

wiki_ppaths_rel = []
for ppath in wiki_ppaths:
    ppath = ppath.replace('/homes/rpujari/scratch/', './data/')
    wiki_ppaths_rel.append(ppath)

print(wiki_ppaths_rel[0])

t1 = datetime.now()

if not os.path.exists(fpath + 'wiki_doc_tensors.pkl'):
    dembs = get_bert_embs(wiki_ppaths_rel, is_parses=False, batch_size=100, max_sent_len=150)
    with open(fpath + 'wiki_doc_tensors.pkl', 'wb') as outfile:
        pickle.dump(dembs, outfile)

t2 = datetime.now()
print('Wiki embeddings computed. Time taken: ', t2 - t1)



# # Background embedding computation

with open(fpath + 'background_ent_mentions.pkl', 'rb') as infile:
    background_parses, bg_eset, bg_sref_dict = pickle.load(infile)

background_parses_rel = []
for ppath in background_parses:
    ppath = ppath.replace('/homes/rpujari/scratch/', './data/')
    background_parses_rel.append(ppath)

t1 = datetime.now()

if not os.path.exists(fpath + 'background_doc_tensors.pkl'):
    dembs = get_bert_embs(background_parses_rel, is_parses=False, batch_size=100, max_sent_len=150)
    with open(fpath + 'background_doc_tensors.pkl', 'wb') as outfile:
        pickle.dump(dembs, outfile)

t2 = datetime.now()
print('Background embeddings computed. Time taken: ', t2 - t1)


# # Statement embedding computation

with open(fpath + 'formatted_statement_data.pkl', 'rb') as infile:
    formatted_statement_data = pickle.load(infile)
with open(fpath + 'stat_ent_mentions.pkl', 'rb') as infile:
    stat_ppaths, p_eset, p_sref_dict = pickle.load(infile)

stat_ppaths_rel = []
for ppath in stat_ppaths:
    ppath = ppath.replace('/homes/rpujari/scratch/', './data/')
    stat_ppaths_rel.append(ppath)

print(stat_ppaths_rel[0])

t1 = datetime.now()

if not os.path.exists(fpath + 'stat_doc_tensors.pkl'):
    dembs = get_bert_embs(stat_ppaths_rel, is_parses=False, batch_size=20, max_sent_len=100)
    with open(fpath + 'stat_doc_tensors.pkl', 'wb') as outfile:
        pickle.dump(dembs, outfile)

if not os.path.exists(fpath + 'stat_doc_tensors_poprem.pkl'):
    stat_popular_entities = [get_popular_entity(p_sref_dict[i], p_eset) for i in range(len(stat_ppaths))]
    dembs = get_bert_embs(stat_ppaths_rel, is_parses=False, batch_size=20, max_sent_len=100, popular_entities=stat_popular_entities)
    with open(fpath + 'stat_doc_tensors_poprem.pkl', 'wb') as outfile:
        pickle.dump((dembs, stat_popular_entities), outfile)

t2 = datetime.now()
print('Statement embeddings computed. Time taken: ', t2 - t1)



