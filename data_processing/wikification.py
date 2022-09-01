#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import os
import sys
from datetime import datetime
import nltk
import threading
import math
import json
import torch
import re
import urllib
from bs4 import BeautifulSoup as BS


# In[2]:


#path to the directory with .txt files
dir_path = '/homes/rpujari/scratch1_fortytwo/compositional_learner_2.0/data/statements/'


#Register and get a wikifier key from https://wikifier.org/register.html
wikifier_api_key = ''

#cuda device
cuda_device = 0


# In[3]:


data_files = os.listdir(dir_path)
data_ppaths = []

for fname in data_files:
    if fname.endswith('.parse'):
        data_ppaths.append(dir_path + fname)



# ## Text Processing Functions

# In[13]:


def get_doc_text(pparse, is_path=True, max_sent_len=150):
    if is_path:
        pparse = json.load(open(pparse, 'r'))
    ret_text = []
    for para in pparse['tokens']:
        for sent in para:
            stext = [tok['originalText'] for tok in sent['tokens'] if tok['originalText'] not in ["\\", "\\\\", 'b', 'r', 'n']][:max_sent_len]
            stext = [re.sub("x[a-fA-F0-9][a-fA-F0-9]", '', tok).strip() for tok in stext]
            stext = [tok for tok in stext if tok]
            if len(stext) > 0:
                ret_text.append(' '.join(stext))
    return ret_text


# In[14]:


def batchify(all_sents, batch_size=100):
    batches = []
    beg = 0
    end = batch_size
    while beg < len(all_sents):
        batches.append(all_sents[beg:end])
        beg = end
        end += batch_size
    return batches


# In[15]:


def create_bert_emb(all_sents, tok_pooling='mean', get_cls_emb=False):
    if len(all_sents) > 0:
        with torch.cuda.device(cuda_device):
            all_toks = tokenizer.batch_encode_plus(all_sents, padding='longest',                                                   add_special_tokens=True)
            tok_tensor = torch.tensor(all_toks['input_ids']).to('cuda')
            attn_mask = torch.tensor(all_tok['attention_mask']).to('cuda')
            tok_tensor = tok_tensor[:, :512]
            with torch.no_grad():
                model_out = model(tok_tensor, attention_mask=attn_mask)
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


# In[16]:


def get_bert_embs(ppaths, is_parses=True, batch_size=100, max_sent_len=150, tok_pooling='sum', get_cls_emb=False):
    ret_embs = []
    t1 = datetime.now()
    all_sents = []
    doc_lens = []
    for i, ppath in enumerate(ppaths):
        if not is_parses:
            doc_text = get_doc_text(ppath, is_path=True, max_sent_len=max_sent_len)
        else:
            doc_text = get_doc_text(ppath, is_path=False, max_sent_len=max_sent_len)
        doc_lens.append(len(doc_text))
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
        batch_tensor = create_bert_emb(batch, tok_pooling=tok_pooling, get_cls_emb=get_cls_emb)
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


# In[17]:


def get_reverse_coref_dict(eg_cref):
    tidx_dict = {}
    for idx, cc in enumerate(eg_cref):
        tidx_dict[idx] = {}
        for key in cc:
            cc_set = cc[key]
            repMention = None
            for j, item in enumerate(cc_set):
                if item['isRepresentativeMention']:
                    repMention = j
                    break
            for k, item in enumerate(cc_set):
                isnum = item['sentNum'] - 1
                for i in range(item['startIndex'] - 1, item['endIndex'] - 1):
                    if not isnum in tidx_dict[idx]:
                        tidx_dict[idx][isnum] = {}
                    tidx_dict[idx][isnum][i] = (key, repMention, k)
    return tidx_dict


# In[33]:


def get_entity_set(ppaths, is_parses=False):
    ent_dict = {}
    sent_ref_dict = {}
    sel_ner_set = ['RELIGION', 'COUNTRY', 'PERSON', 'ORGANIZATION', 'CITY', 'STATE_OR_PROVINCE', 'IDEOLOGY']
    t1 = datetime.now()
    tot_path_num = len(ppaths)
    #For each path in the list of paths
    for path_num, ppath in enumerate(ppaths):
        t2 = datetime.now()
        # print(path_num)
        if (path_num + 1) % 100 == 0:
            print(path_num, '/', tot_path_num, 'done', t2-t1)
            print('Estimated finish time', datetime.now() + (t2-t1) * ((tot_path_num - path_num) / path_num))
        if is_parses:
            eg = ppath
        else:
            eg = json.load(open(ppath))
        
        sent_ref_dict[path_num] = {}
        #Build reverse co-ref dictionary (token index -> head entity) for the document
        cref_rev = get_reverse_coref_dict(eg['coref'])
        
        #For each para in text
        for pnum in range(len(eg['tokens'])):
            eg_para = eg['tokens'][pnum]
            ent_ref_dict = {}
            sent_ref_dict[path_num][pnum] = {}
            
            #For each sentence in the para
            for sidx in range(len(eg_para)):
                #Get entity mentions
                eg_ents = eg_para[sidx]['entitymentions']
                ent_ref_dict[sidx] = {}
                #For each mention of entity, label the token index with the entity mention index
                for i, eg_ent in enumerate(eg_ents):
                    for j in range(eg_ent['tokenBegin'], eg_ent['tokenEnd']):
                        ent_ref_dict[sidx][j] = i
    
            #For each sentence in the para            
            for sidx in range(len(eg_para)):
                eg_ents = eg_para[sidx]['entitymentions']
                sent_ref_dict[path_num][pnum][sidx] = {}
                #For each entity mention in the sentence
                for eg_ent in eg_ents:
                    #If entity mentions's NER tag is in desired tag set
                    if eg_ent['ner'] in sel_ner_set:
                        #Get entity mention's text and its POS
                        ent_text = eg_ent['text']
                        ent_pos = eg_para[sidx]['tokens'][eg_ent['tokenBegin']]['pos']
                        
                        #For each para in reverse coref dictionary
                        if pnum in cref_rev:
                            #If the current sentence is in the reverse co-ref dictionary
                            if sidx in cref_rev[pnum]:
                                #If beginning token of the entity mention has a head entity mapping
                                if eg_ent['tokenBegin'] in cref_rev[pnum][sidx]:
                                    key, rm, k = cref_rev[pnum][sidx][eg_ent['tokenBegin']]
                                    rm = eg['coref'][pnum][key][rm]
                                    rm_sidx = rm['sentNum'] - 1
                                    rm_tidx = rm['headIndex'] - 1
                                    
                                    #If coref unit head index has a token -> entity mention mapping, then update ent_text
                                    if rm_tidx in ent_ref_dict[rm_sidx]: 
                                        rm_eidx = ent_ref_dict[rm_sidx][rm_tidx]
                                        rm_ent = eg_para[rm_sidx]['entitymentions'][rm_eidx]
                                        ent_text = rm_ent['text']
                                    #Else, just assign text of the coref unit head
                                    else:
                                        ent_text = rm['text']
                                    rm_head = eg_para[rm['sentNum'] - 1]['tokens'][rm['headIndex'] - 1]
                                    ent_pos = rm_head['pos']

                        #If entity mention POS is not a pronoun, then add it to the enity dictionary
                        if not ent_pos.startswith('PRP'):
                            if ent_text in ent_dict:
                                ent_dict[ent_text] += 1
                            else:
                                ent_dict[ent_text] = 1
                            #Add the entity text to the sentence's entity list
                            if ent_text in sent_ref_dict[path_num][pnum][sidx]:
                                sent_ref_dict[path_num][pnum][sidx][ent_text] += 1
                            else:
                                sent_ref_dict[path_num][pnum][sidx][ent_text] = 1
    return ent_dict, sent_ref_dict


# In[19]:


def get_formatted_data(wparse, ment_dict):
    ret_text = []
    for para in wparse['tokens']:
        for sent in para:
            stext = ' '.join([tok['originalText'] for tok in sent['tokens']])
            ret_text.append(stext)
    ret_ents = []
    for para in ment_dict:
        for sent in ment_dict[para]:
            s_ents = []
            for ent in ment_dict[para][sent]:
                if ent in wiki_ann_dict:
                    s_ents.append(ent)
            ret_ents.append(s_ents)
    return ret_text, ret_ents


# In[20]:


def concat_doc_tensors(d_tensors):
    s_len = max([t.size(1) for t in d_tensors])
    emb_size = d_tensors[0].size(-1)
    ret_tensor = torch.FloatTensor(len(d_tensors), s_len, emb_size)
    ret_mask = torch.zeros(len(d_tensors), s_len)
    for i, d_tensor in enumerate(d_tensors):
        ret_tensor[i, :d_tensor.size(1), :] = d_tensor
        ret_mask[i, :d_tensor.size(1)] = 1
    return ret_tensor, ret_mask



all_ppaths = []
fnames = os.listdir(dir_path)
for fname in fnames:
    if fname.endswith('.parse'):
        all_ppaths.append(dir_path + fname)
print(len(all_ppaths))


# ### Entity Extraction and Wikification

eset, sr_set = get_entity_set(all_ppaths)


# In[34]:


c = 0
sel_ents = []
for ent in eset:
    if len(ent.split()) <= 6 and eset[ent] >= 25:
        sel_ents.append(ent)
#         print(ent)
        c += 1
print(c)


# In[ ]:


def call_wikifier(text, lang="en", threshold=0.6):
    global wiki_ann_dict
    # Prepare the URL.
    data = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", wikifier_api_key),
#         ("pageRankSqThreshold", "%g" % threshold), ("applyPageRankSqThreshold", "true"),
#         ("nTopDfValuesToIgnore", "200"), ("nWordsToIgnoreFromList", "200"),
#         ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
#         ("support", "true"), ("ranges", "false"),
#         ("includeCosines", "false"), ("maxMentionEntropy", "3")
        ])
    url = "http://www.wikifier.org/annotate-article"
    
    # Call the Wikifier and read the response.
    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    with urllib.request.urlopen(req, timeout = 60) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))

    # Sort annotations by cosine score and store in dict 
    if len(response['annotations']) > 0:
        sorted_ann_list = sorted(response['annotations'], key=lambda x:x['cosine'], reverse=True)
        wiki_ann_dict[text] = sorted_ann_list


# In[ ]:


def wikify_entities(sel_ent_list, batch_size=10):
    global wiki_ann_dict
    tlen = len(sel_ent_list)
    i = 0
    bnum = 0
    num_batches = math.ceil(tlen / batch_size)
    while i < tlen:
        batch = sel_ent_list[i : i+batch_size]
        bthreads = []
        for ent in batch:
            tthread = threading.Thread(target=call_wikifier, args=(ent,))
            tthread.start()
            bthreads.append(tthread)
        for tthread in bthreads:
            tthread.join()
        i += batch_size
        bnum += 1
        if bnum%100 == 0 and bnum > 0:
            print(bnum, ' / ', num_batches, ' done.')


# In[ ]:


global wiki_ann_dict
wiki_ann_dict = {}
wikify_entities(sel_ents)


# In[ ]:


title_dict = {}
title_set = set()
for ent in wiki_ann_dict:
    title_dict[ent] = wiki_ann_dict[ent][0]['title']
    title_set.add(wiki_ann_dict[ent][0]['title'])


# In[ ]:


fnames = [x.split('/')[-1][:-6] for x in all_ppaths]
with open(dir_path + 'extracted_entities.pkl', 'wb') as outfile:
    pickle.dump((fnames, eset, sr_set, title_dict, title_set), outfile)
    
with open(dir_path + 'wiki_ann_dict.pkl', 'wb') as outfile:
    pickle.dump(wiki_ann_dict, outfile)



