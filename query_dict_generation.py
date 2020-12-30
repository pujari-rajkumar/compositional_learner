#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import random
from datetime import datetime


# In[3]:


from query_handling import query_processing
from query_handling import data_splitting_utils


# In[4]:


def generate_lp_data(queries, seed=4056):
    random.seed(seed)
    query_dicts = []
    qd_splits = []
    doc_lists = []
    start_time = datetime.now()
    for i, q in enumerate(queries):
        ret_dict = query_processing.get_data(q[0], q[1], q[2])
        t, d, e = data_splitting_utils.create_data_splits(ret_dict, seed=seed)
        query_dicts.append(ret_dict)
        qd_splits.append((t, d, e))
        time_now = datetime.now()
        print(i, ' done - ', time_now - start_time)
    
    return query_dicts, qd_splits


# In[5]:


with open('./data/composite_learner_data/entity_name_dict.pkl', 'rb') as infile:
    entity_name_dict = pickle.load(infile)


# In[6]:


import os
topics = os.listdir('./data/all_sides/news/')


# In[7]:


n = 0
t = 0
queries = []
for wlink in entity_name_dict:
    en_dict = entity_name_dict[wlink]
    nt = 0
    sname = ''
    for dtype in en_dict:
        if len(en_dict[dtype]) > 0:
            nt += 1
            sname = en_dict[dtype][0]
    if nt >= 4:
        for topic in topics:
            q = ([sname], [topic], [[-1]])
            queries.append(q)


# In[8]:


print(len(queries), 'queries generated.')


# In[10]:


with open('./data/composite_learner_data/data_examples/queries.pkl', 'wb') as outfile:
    pickle.dump(queries, outfile)


# In[11]:


bsz = 24
bnum = 0
b = 0
fpath = './data/composite_learner_data/'
t1 = datetime.now()
while b < len(queries):
    e = b + bsz
    dpath = fpath + 'data_examples/query_dicts-' + str(bnum) + '.pkl'
    if not os.path.exists(dpath):
        batch = queries[b:e]
        qds, qd_splits = generate_lp_data(batch)
        with open(dpath, 'wb') as outfile:
            pickle.dump((qds, qd_splits), outfile)
        del qds
        del qd_splits
        t2 = datetime.now()
        print('Batch ', bnum, 'done - ', t2-t1, '\n')
    else:
        print('Batch ', bnum, 'done - ', t2-t1, '\n')
    bnum += 1
    b = e


# In[ ]:




