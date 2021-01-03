#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import os
import sys
from datetime import datetime
from stanfordcorenlp import StanfordCoreNLP
import nltk
import threading
import math
import json


# In[2]:


def process_article(tpath):
    global proc_dict
    try:
        ppath = tpath[:-3] + 'parse'
        if not os.path.exists(ppath):
            ttext = open(tpath).read()
            tsplit = ttext.split('\n')
            tparas = []
            tsents = []
            tcoref = []
            for tpara in tsplit:
                if tpara.strip() != '':
                    tparas.append(tpara.strip())
                    ptoks = json.loads(stanford_nlp.annotate(tpara.strip(), properties={'annotators': 'ssplit,pos,coref,ner', 'outputFormat': 'json'}))
                    tsents.append(ptoks['sentences'])
                    tcoref.append(ptoks['corefs'])
            tparse = {'text': tparas, 'tokens': tsents, 'coref': tcoref}
            json.dump(tparse, open(ppath, 'w'))
    except:
        print(sys.exc_info()[0], tpath)


# In[3]:


def process_all_articles(tpaths, batch_size=10): 
    b1 = 0
    b2 = batch_size
    bnum = 0
    tot_batches = math.ceil(len(tpaths) / batch_size)
    while b1 < len(tpaths):
        t1 = datetime.now()
        bpaths = tpaths[b1:b2]
        bthreads = []
        for tpath in bpaths:
            if os.path.exists(tpath):
                tthread = threading.Thread(target=process_article, args=(tpath,))
                tthread.start()
                bthreads.append(tthread)
        for bthread in bthreads:
            bthread.join()
        b1 += batch_size
        b2 += batch_size
        t2 = datetime.now()
        bnum += 1
        print(bnum, '/', tot_batches, ' batches done. Time taken: ', t2-t1)


# In[ ]:


stanford_nlp = StanfordCoreNLP('./stanford-corenlp/')


# In[ ]:

all_text_paths = []

#Collecting news text paths
news_path = './data/all_sides/news/'
issues = os.listdir(news_path)
for issue in issues:
    ifiles = os.listdir(news_path + issue)
    for ifile in ifiles:
        ifpath = news_path + issue + '/' + ifile
        if ifpath.endswith('.txt'):
            all_text_paths.append(ifpath)


#Collecting wiki text paths
wiki_path = './data/on_the_issues/candidate_wiki_pages/'
ifiles = os.listdir(wiki_path)
for ifile in ifiles:
    ifpath = wiki_path + ifile
    if ifpath.endswith('.txt'):
        all_text_paths.append(ifpath)


#Collecting quote text paths
quote_path = './data/on_the_issues/quote_texts/'
issues = os.listdir(quote_path)
for issue in issues:
    ifiles = os.listdir(quote_path + issue)
    for ifile in ifiles:
        ifpath = quote_path + issue + '/' + ifile
        if ifpath.endswith('.txt'):
            all_text_paths.append(ifpath)

#Collecting background text paths
bg_path = './data/on_the_issues/background_texts/'
ifiles = os.listdir(bg_path)
for ifile in ifiles:
    ifpath = bg_path + ifile
    if ifpath.endswith('.txt'):
        all_text_paths.append(ifpath)

#Collecting statement text paths
stat_path = './data/pro_publica/statements/'
issues = os.listdir(stat_path)
for issue in issues:
    ifiles = os.listdir(stat_path + issue)
    for ifile in ifiles:
        ifpath = stat_path + issue + '/' + ifile
        if ifpath.endswith('.txt'):
            all_text_paths.append(ifpath)


print('Collected', len(all_text_paths), 'paths.')

process_all_articles(all_text_paths)

stanford_nlp.close()

