#!/usr/bin/env python
# coding: utf-8

#@author: Rajkumar Pujari

# In[2]:


import pickle
import random


# In[18]:


def split_dict(data, train, dev, test, dtype):
    if dtype in data:
        train[dtype] = {}
        dev[dtype] = {}
        test[dtype] = {}
        for topic in data[dtype][0]:
            train[dtype][topic] = {}
            dev[dtype][topic] = {}
            test[dtype][topic] = {}
            for event in data[dtype][0][topic]:
                toss = random.random()
                if toss <= 0.8:
                    train[dtype][topic][event] = []
                    dev[dtype][topic][event] = []
                    test[dtype][topic][event] = []
                    for i, doc in enumerate(data[dtype][0][topic][event]):
                        dtoss = random.random()
                        if dtoss <= 0.8:
                            train[dtype][topic][event].append(i)
                        elif dtoss > 0.8 and dtoss < 0.9:
                            dev[dtype][topic][event].append(i)
                        else:
                            test[dtype][topic][event].append(i)
                elif toss > 0.8 and toss < 0.9:
                    dev[dtype][topic][event] = list(range(len(data[dtype][0][topic][event])))
                else:
                    test[dtype][topic][event] = list(range(len(data[dtype][0][topic][event])))
    return train, dev, test


# In[19]:

def create_data_splits(data_dict, seed):
    random.seed(seed)
    train_dict = {}
    dev_dict = {}
    test_dict = {}
    for ent in data_dict:
        if ent == 'description':
            continue
        elif ent == 'news':
            train_dict, dev_dict, test_dict = split_dict(data_dict, train_dict, dev_dict, test_dict, 'news')
        else:
            train_dict[ent] = {}
            dev_dict[ent] = {}
            test_dict[ent] = {}

            if 'tweets' in data_dict[ent]:
                train_dict[ent], dev_dict[ent], test_dict[ent] = split_dict(data_dict[ent], train_dict[ent], dev_dict[ent], test_dict[ent], 'tweets')

            if 'statements' in data_dict[ent]:
                train_dict[ent], dev_dict[ent], test_dict[ent] = split_dict(data_dict[ent], train_dict[ent], dev_dict[ent], test_dict[ent], 'statements')
                
    return train_dict, dev_dict, test_dict


def fill_data(orig_dict, ret_dict, comb_idx_dict, idx_dict, key):
    if key not in ret_dict:
        ret_dict[key] = ({}, {}, {})
        comb_idx_dict[key] = {}
    for topic in idx_dict[key]:
        if topic not in ret_dict[key][0]:
            ret_dict[key][0][topic] = {}
            ret_dict[key][1][topic] = {}
            ret_dict[key][2][topic] = {}
            comb_idx_dict[key][topic] = {}
        for event in idx_dict[key][topic]:
            if event not in ret_dict[key][0][topic]:
                ret_dict[key][0][topic][event] = []
                ret_dict[key][1][topic][event] = []
                ret_dict[key][2][topic][event] = []
                comb_idx_dict[key][topic][event] = []
            for idx in idx_dict[key][topic][event]:
                ret_dict[key][0][topic][event].append(orig_dict[key][0][topic][event][idx])
                ret_dict[key][1][topic][event].append(orig_dict[key][1][topic][event][idx])
                ret_dict[key][2][topic][event].append(orig_dict[key][2][topic][event][idx])
                comb_idx_dict[key][topic][event].append(idx)
    return ret_dict, comb_idx_dict


def get_documents(orig_dict, idx_dicts):
    ret_dict = {}
    comb_idx_dict = {}
    if 'description' in orig_dict:
        ret_dict['description'] = orig_dict['description']
    for idx_dict in idx_dicts:
        for ent in idx_dict:
            if ent not in ['news'] and ent not in ret_dict:
                ret_dict[ent] = {}
                comb_idx_dict[ent] = {}
                if 'wiki' in orig_dict[ent]:
                    ret_dict[ent]['wiki'] = orig_dict[ent]['wiki']
                if 'quotes' in orig_dict[ent]:
                    ret_dict[ent]['quotes'] = orig_dict[ent]['quotes']
        ret_dict, comb_idx_dict = fill_data(orig_dict, ret_dict, comb_idx_dict, idx_dict, 'news')
        for ent in idx_dict:
            if ent not in ['news']:
                if 'tweets' in idx_dict[ent]:
                    ret_dict[ent], comb_idx_dict[ent] = fill_data(orig_dict[ent], ret_dict[ent], comb_idx_dict[ent], idx_dict[ent], 'tweets')
                if 'statements' in idx_dict[ent]:
                    ret_dict[ent], comb_idx_dict[ent] = fill_data(orig_dict[ent], ret_dict[ent], comb_idx_dict[ent], idx_dict[ent], 'statements')
    return ret_dict, comb_idx_dict
                
        
