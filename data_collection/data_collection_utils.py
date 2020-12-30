#!/usr/bin/env python
# coding: utf-8

# @author: Rajkumar Pujari

#Import Libraries
import time
import urllib.request
import re
from bs4 import BeautifulSoup as BS
import os
import unidecode


#Downloading entire Web Document (Raw Page Content)
def download_page(url):
    try:
        headers = {}
        headers['User-Agent'] = "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"
        req = urllib.request.Request(url, headers = headers)
        resp = urllib.request.urlopen(req)
        respData = str(resp.read())
        return respData
    except Exception as e:
        print(str(e))


def split_sents(text_path):
    from nltk.tokenize import sent_tokenize, word_tokenize
    fname = text_path.split('/')[-1][:-5]
    dirpath = '/'.join(text_path.split('/')[:-1])
    with open(text_path) as infile:
        with open(dirpath + '/' + fname + '.txt', 'w') as outfile:
            lt_soup = BS(infile.read(), 'html.parser')
            paras = lt_soup.find_all('p')
            for para in paras:
                if para.text.strip():
                    ptext = re.sub('\[[a-z 0-9]+\]', '', para.text.strip())
                    ptext = ptext.replace('\\n', '').strip()
                    if len(ptext) > 5:
                        ptoks = word_tokenize(ptext)
                        tok_para = ' '.join(ptoks)
                        outfile.write(tok_para + '\n')


