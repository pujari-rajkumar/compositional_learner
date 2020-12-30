#!/usr/bin/env python
# coding: utf-8

#@author: Rajkumar Pujari    

# In[1]:


import pickle
import json
import os
import math
import urllib.parse
import urllib.request
import threading
import torch


# In[2]:


fpath = './data/composite_learner_data/'
wpath = './data/on_the_issues/candidate_wiki_pages/'


# In[3]:


with open(fpath + 'wiki_ann_dict.pkl', 'rb') as infile:
    wiki_ann_dict = pickle.load(infile)


# In[4]:


with open(fpath + 'formatted_tweet_parses.pkl', 'rb') as infile:
    formatted_tweet_parses = pickle.load(infile)
with open(fpath + 'formatted_tweet_ent_mentions.pkl', 'rb') as infile:
    _, ft_esets, ft_sref_dict = pickle.load(infile)
with open(fpath + 'tweet_doc_tensors.pkl', 'rb') as infile:
    tweet_doc_tensors = pickle.load(infile)



# In[5]:


with open(fpath + 'formatted_quote_data.pkl', 'rb') as infile:
    formatted_quote_data = pickle.load(infile)
with open(fpath + 'quote_ent_mentions.pkl', 'rb') as infile:
    quote_ppaths, q_eset, q_sref_dict = pickle.load(infile)
with open(fpath + 'quote_doc_tensors.pkl', 'rb') as infile:
    quote_doc_tensors = pickle.load(infile)

# In[6]:


with open(fpath + 'formatted_statement_data.pkl', 'rb') as infile:
    formatted_statement_data = pickle.load(infile)
with open(fpath + 'stat_ent_mentions.pkl', 'rb') as infile:
    stat_ppaths, p_eset, p_sref_dict = pickle.load(infile)
with open(fpath + 'stat_doc_tensors.pkl', 'rb') as infile:
    stat_doc_tensors = pickle.load(infile)

# In[7]:


with open(fpath + 'formatted_news_data.pkl', 'rb') as infile:
    formatted_news_data = pickle.load(infile)
with open(fpath + 'news_ent_mentions.pkl', 'rb') as infile:
    news_ppaths, n_eset, n_sref_dict = pickle.load(infile)
with open(fpath + 'news_doc_tensors.pkl', 'rb') as infile:
    news_doc_tensors = pickle.load(infile)


# In[8]:


with open(fpath + 'wiki_ent_mentions.pkl', 'rb') as infile:
    wiki_ppaths, w_eset, w_sref_dict = pickle.load(infile)
with open(fpath + 'wiki_doc_tensors.pkl', 'rb') as infile:
    wiki_doc_tensors = pickle.load(infile)


# In[10]:


with open(fpath + 'background_ent_mentions.pkl', 'rb') as infile:
    background_parses, bg_eset, bg_sref_dict = pickle.load(infile)
with open(fpath + 'background_doc_tensors.pkl', 'rb') as infile:
    background_doc_tensors = pickle.load(infile)

# In[11]:


with open(fpath + 'entity_wlink_dict.pkl', 'rb') as infile:
    entity_wlink_dict = pickle.load(infile)
with open(fpath + 'entity_name_dict.pkl', 'rb') as infile:
    entity_name_dict = pickle.load(infile)


# #Data fetching methods

# In[17]:


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


# In[18]:


def get_tweets(entity, topics, events):
    ret_docs = {}
    ret_ents = {}
    ret_embs = {}
    if entity in formatted_tweet_parses:
        etopics = formatted_tweet_parses[entity]
        for i, topic in enumerate(topics):
            if topic in etopics:
                ret_docs[topic] = {}
                ret_ents[topic] = {}
                ret_embs[topic] = {}
                tevents = events[i]
                if tevents[0] == -1:
                    tevents = list(etopics[topic].keys())
                for event in tevents:
                    if event in etopics[topic]:
                        ret_docs[topic][event] = []
                        ret_ents[topic][event] = []
                        ret_embs[topic][event] = []
                        for tweet in etopics[topic][event]:
                            tparse = tweet['parse']
                            ment_dict = ft_sref_dict[tweet['parse_id']]
                            ret_tensor = tweet_doc_tensors[tweet['parse_id']]
                            ret_text, ret_ent = get_formatted_data(tparse, ment_dict)
                            if len(ret_text) > 0:
                                ret_docs[topic][event].append(ret_text)
                                ret_ents[topic][event].append(ret_ent)
                                ret_embs[topic][event].append(ret_tensor)
    return ret_docs, ret_ents, ret_embs


# In[19]:


def get_statements(entity, topics, events):
    ppath = './data/pro_publica/statements/'
    ret_docs = {}
    ret_ents = {}
    ret_embs = {}
    if entity in formatted_statement_data:
        edata = formatted_statement_data[entity]
        for i, topic in enumerate(topics):
            if topic in edata:
                ret_docs[topic] = {}
                ret_ents[topic] = {}
                ret_embs[topic] = {}
                tevents = events[i]
                if tevents[0] == -1:
                    tevents = list(edata[topic].keys())
                for event in tevents:
                    if event in edata[topic]:
                        ret_docs[topic][event] = []
                        ret_ents[topic][event] = []
                        ret_embs[topic][event] = []
                        for statement in edata[topic][event]:
                            sid = statement['statement_id']
                            spath = ppath + topic + '/statement_' + str(sid) + '.parse'
                            if spath in stat_ppaths:
                                sidx = stat_ppaths.index(spath)
                                sparse = json.load(open(spath, 'r'))
                                ment_dict = p_sref_dict[sidx]
                                ret_tensor = stat_doc_tensors[sidx]
                                ret_text, ret_ent = get_formatted_data(sparse, ment_dict)
                                if len(ret_text) > 0:
                                    ret_docs[topic][event].append(ret_text)
                                    ret_ents[topic][event].append(ret_ent)
                                    ret_embs[topic][event].append(ret_tensor)
    return ret_docs, ret_ents, ret_embs


# In[26]:


def get_news(topic, events):
    ret_docs = {}
    ret_ents = {}
    ret_embs = {}
    apath = './data/all_sides/news/'
    if topic in formatted_news_data:
        ret_docs[topic] = {}
        ret_ents[topic] = {}
        ret_embs[topic] = {}
        ndata = formatted_news_data[topic]
        nevents = events
        if nevents[0] == -1:
            nevents = list(ndata.keys())
        for event in nevents:
            if event in ndata:
                ret_docs[topic][event] = []
                ret_ents[topic][event] = []
                ret_embs[topic][event] = []
                for article in ndata[event]:
                    ppath = apath + topic + '/' + article['id'] + '.parse'
                    if ppath in news_ppaths:
                        nidx = news_ppaths.index(ppath)
                        nparse = json.load(open(ppath, 'r'))
                        ment_dict = n_sref_dict[nidx]
                        ret_tensor = news_doc_tensors[nidx]
                        ret_text, ret_ent = get_formatted_data(nparse, ment_dict)
                        if len(ret_text) > 0:
                            ret_docs[topic][event].append(ret_text)
                            ret_ents[topic][event].append(ret_ent)
                            ret_embs[topic][event].append(ret_tensor)
    return ret_docs, ret_ents, ret_embs


# In[21]:


def get_quotes(entity, topics):
    ret_docs = {}
    ret_ents = {}
    ret_embs = {}
    qpath = './data/on_the_issues/quote_texts/'
    for topic in topics:
        ppath = qpath + topic + '/' + entity.replace(' ', '_') + '.parse'
        if ppath in quote_ppaths:
            ret_docs[topic] = []
            ret_ents[topic] = []
            ret_embs[topic] = []
            pidx = quote_ppaths.index(ppath)
            qparse = json.load(open(ppath, 'r'))
            ment_dict = q_sref_dict[pidx]
            ret_tensor = quote_doc_tensors[pidx]
            ret_text, ret_ent = get_formatted_data(qparse, ment_dict)
            if len(ret_text) > 0:
                ret_docs[topic].append(ret_text)
                ret_ents[topic].append(ret_ent)
                ret_embs[topic].append(ret_tensor)
    return ret_docs, ret_ents, ret_embs


# In[22]:


def get_wiki(entity):
    ppath = wpath + entity.replace(' ', '_') + '.parse'
    ret_docs = []
    ret_ents = []
    ret_embs = []
    if ppath in wiki_ppaths:
        wparse = json.load(open(ppath))
        pidx = wiki_ppaths.index(ppath)
        ment_dict = w_sref_dict[pidx]
        ret_tensor = wiki_doc_tensors[pidx]
        ret_text, ret_ent = get_formatted_data(wparse, ment_dict)
        if len(ret_text) > 0:
            ret_docs.append(ret_text)
            ret_ents.append(ret_ent)
            ret_embs.append(ret_tensor)
    return ret_docs, ret_ents, ret_embs


# In[23]:


def get_description(topic):
    dpath = './data/on_the_issues/background_texts/' + topic + '_background.parse'
    ret_docs = []
    ret_ents = []
    ret_embs = []
    if os.path.exists(dpath):
        dparse = json.load(open(dpath, 'r'))
        didx = background_parses.index(dpath)
        ment_dict = bg_sref_dict[didx]
        ret_tensor = background_doc_tensors[didx]
        ret_text, ret_ent = get_formatted_data(dparse, ment_dict)
        if len(ret_text) > 0:
            ret_docs.append(ret_text)
            ret_ents.append(ret_ent)
            ret_embs.append(ret_tensor)
    return ret_docs, ret_ents, ret_embs


# In[73]:


def get_data(entities, topics, events):
    ret_data = {}
    
    #Get news for the events
    ret_data['news'] = ({}, {}, {})
    for j, topic in enumerate(topics):
        if topic in formatted_news_data:
            n_docs, n_ents, n_embs = get_news(topic, events[j])
            c_d, c_e, c_t = ret_data['news']
            for topic in n_docs:
                if topic not in c_d:
                    c_d[topic] = n_docs[topic]
                    c_e[topic] = n_ents[topic]
                    c_t[topic] = n_embs[topic]
                else:
                    for event in n_docs[topic]:
                        if event not in c_d[topic]:
                            c_d[topic][event] = n_docs[topic][event]
                            c_e[topic][event] = n_ents[topic][event]
                            c_t[topic][event] = n_embs[topic][event]
                        else:
                            c_d[topic][event] += n_docs[topic][event]
                            c_e[topic][event] += n_ents[topic][event]
                            c_t[topic][event] += n_embs[topic][event]
            ret_data['news'] = (c_d, c_e, c_t)
    
    #Get descriptions of topics
    ret_data['description'] = ({}, {}, {})
    for j, topic in enumerate(topics):
        d_docs, d_ents, d_embs = get_description(topic)
        c_d, c_e, c_t = ret_data['description']
        if len(d_docs) > 0:
            c_d[topic] = d_docs
            c_e[topic] = d_ents
            c_t[topic] = d_embs
        ret_data['description'] = (c_d, c_e, c_t)
        
    #For each entity in the query
    for i, entity in enumerate(entities):
        e_w_link = entity_wlink_dict[entity]
        e_name_dict = entity_name_dict[e_w_link]
        ret_data[entity] = {}
        
        #Fetch entity's tweets
        if len(e_name_dict['tweets'] ) > 0:
            ret_data[entity]['tweets'] = ({}, {}, {})
            for name in e_name_dict['tweets']:
                t_docs, t_ents, t_embs = get_tweets(name, topics, events)
                c_d, c_e, c_t = ret_data[entity]['tweets']
                for topic in t_docs:
                    if topic not in c_d:
                        c_d[topic] = t_docs[topic]
                        c_e[topic] = t_ents[topic]
                        c_t[topic] = t_embs[topic]
                    else:
                        for event in t_docs[topic]:
                            if event not in c_d[topic]:
                                c_d[topic][event] = t_docs[topic][event]
                                c_e[topic][event] = t_ents[topic][event]
                                c_t[topic][event] = t_embs[topic][event]
                            else:
                                c_d[topic][event] += t_docs[topic][event]
                                c_e[topic][event] += t_ents[topic][event]
                                c_t[topic][event] += t_embs[topic][event]
                ret_data[entity]['tweets'] = (c_d, c_e, c_t)
        
        #Fetch entity's statements
        if len(e_name_dict['statements'] ) > 0:
            ret_data[entity]['statements'] = ({}, {}, {})
            for name in e_name_dict['statements']:
                s_docs, s_ents, s_embs = get_statements(name, topics, events)
                c_d, c_e, c_t = ret_data[entity]['statements']
                for topic in s_docs:
                    if topic not in c_d:
                        c_d[topic] = s_docs[topic]
                        c_e[topic] = s_ents[topic]
                        c_t[topic] = s_embs[topic]
                    else:
                        for event in s_docs[topic]:
                            if event not in c_d[topic]:
                                c_d[topic][event] = s_docs[topic][event]
                                c_e[topic][event] = s_ents[topic][event]
                                c_t[topic][event] = s_embs[topic][event]
                            else:
                                c_d[topic][event] += s_docs[topic][event]
                                c_e[topic][event] += s_ents[topic][event]
                                c_t[topic][event] += s_embs[topic][event]
                ret_data[entity]['statements'] = (c_d, c_e, c_t)
        
        #Fetch entity's quotes
        if len(e_name_dict['quotes'] ) > 0:
            ret_data[entity]['quotes'] = ({}, {}, {})
            for name in e_name_dict['quotes']:
                q_docs, q_ents, q_embs = get_quotes(name, topics)
                c_d, c_e, c_t = ret_data[entity]['quotes']
                for topic in q_docs:
                    if topic not in c_d:
                        c_d[topic] = q_docs[topic]
                        c_e[topic] = q_ents[topic]
                        c_t[topic] = q_embs[topic]
                    else:
                        c_d[topic] += q_docs[topic]
                        c_e[topic] += q_ents[topic]
                        c_t[topic] += q_embs[topic]
                ret_data[entity]['quotes'] = (c_d, c_e, c_t)
        
        #Fetch entity's wiki page
        if len(e_name_dict['wiki'] ) > 0:
            ret_data[entity]['wiki'] = ([], [], [])
            for name in e_name_dict['wiki']:
                w_docs, w_ents, w_embs = get_wiki(name)
                c_d, c_e, c_t = ret_data[entity]['wiki']
                ret_data[entity]['wiki'] = (c_d + w_docs, c_e + w_ents, c_t + w_embs)
    
    return ret_data

