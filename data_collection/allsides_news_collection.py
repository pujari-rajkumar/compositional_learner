#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import requests
import pickle
from bs4 import BeautifulSoup as BS
import numpy as np
import threading
import sys


# In[22]:


fpath = './data/all_sides_updated/'


# In[2]:


orig_url_dict = {}
written_articles = 0
with open(fpath + 'clean_url_dict.pkl', 'rb') as infile:
    clean_url_dict = pickle.load(infile)


# In[3]:


global written_articles
written_articles = 0


# In[19]:


def get_write_page(url, fname):
#     print('Starting', fname, url)
    global written_articles
    try:
        t = requests.get(url=url, timeout=15)
        
        if t.text:
            with open(fname, 'w') as f1:
                f1.write(t.text)
                written_articles += 1
    except:
        print(sys.exc_info()[0], url, fname)
#     print(fname, 'done.')


# In[10]:


def get_articles(url_dict, batch_size=100):
    for topic in url_dict:
        urls = url_dict[topic]
        b1 = 0
        b2 = batch_size
        i = 0
        while b1 < len(urls):
            print('Starting', topic, 'batch', i)
            url_threads = []
            for url in urls[b1:b2]:
                fname = fpath + 'news/' + topic + '/' + url['id'] + '.html' 
                if not os.path.exists(fname):
                    uthread = threading.Thread(target=get_write_page, args=(url['url'], fname,))
                    uthread.start()
                    url_threads.append(uthread)
            print('Waiting for', len(url_threads))
            for uthread in url_threads:
                uthread.join()
            b1 += batch_size
            b2 += batch_size
            i += 1
            print(i, 'batch(es) done in topic', topic)



# In[21]:


get_articles(clean_url_dict, batch_size=1000)


# In[23]:


t = 0
for topic in clean_url_dict:
    files = os.listdir(fpath + 'news/' + topic)
    t += len(files)
print(t)


# In[24]:


for topic in clean_url_dict:
    tdir = fpath + 'news/' + topic + '/'
    turls = clean_url_dict[topic]
    for turl in turls:
        thname = tdir + turl['id'] + '.html'
        ttname = tdir + turl['id'] + '.txt'
        if os.path.exists(thname):
            with open(ttname, 'w') as outfile:
                tsoup = BS(open(thname).read())
                tparas = tsoup.find_all('p')
                for tpara in tparas:
                    if tpara.text.strip() != '':
                        outfile.write(tpara.text.strip() + '\n')
    print(topic, ' done.')

