#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pickle
import json
from bs4 import BeautifulSoup as BS
import os
from datetime import date
from data_collection_utils import download_page

# In[2]:


fpath = './data/pro_publica/'
if not os.path.exists(fpath):
    os.mkdir(fpath)


# In[3]:


with open(fpath + 'statement_url_dict.pkl', 'rb') as infile:
    url_dict = pickle.load(infile)


for topic in url_dict:
    t_path = fpath + 'statements/' + topic + '/'
    if not os.path.exists(t_path):
        os.mkdir(t_path)
    for resp in url_dict[topic]:
        if not 'results' in resp:
            continue
        for res in resp['results']:
            i = res['statement_id']
            st_url = res['url']
            if not os.path.exists(t_path + 'statement_' + str(i) + '.html'):
                st_resp = download_page(st_url)
                if st_resp:
                    with open(t_path + 'statement_' + str(i) + '.html', 'w') as outfile:
                        outfile.write(st_resp)
        print('Statements collected: ', i)
    print(topic, ' done. Statements collected: ', i)


# In[16]:

t = 0
c = 0
for topic in url_dict:
    for resp in url_dict[topic]:
        if 'results' in resp:
            for res in resp['results']:
                st_id = res['statement_id']
                st_stub = fpath + topic + '/statement_' + str(st_id)
                if os.path.exists(st_stub + '.html'):
                    st_html = open(st_stub + '.html').read()
                    st_soup = BS(st_html, 'html.parser')
                    st_text = st_soup.find_all('p')
                    st_paras = []
                    for st_para in st_text:
                        st_paras.append(st_para.text.replace('\\n', '').replace('\\t', '').replace('\\b', '').replace('\n', ' '))
                    with open(st_stub + '.txt', 'w') as outfile:
                        for st_para in st_paras:
                            if st_para.strip():
                                outfile.write(st_para.strip() + '\n')
                    c += 1
                t += 1


