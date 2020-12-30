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
from data_collection_utils import download_page, split_sents
                        
#Download senate and house data
fpath = './data/wiki_data/'

if not os.path.exists(fpath):
    os.mkdir(fpath)

#Defining pages
starting_page = "https://en.wikipedia.org/wiki/List_of_current_United_States_senators"
seed_page = "https://en.wikipedia.org"


#Extracting senator data
us_sen_html = download_page(starting_page)
us_sen_soup = BS(us_sen_html, 'html.parser')
sen_table = us_sen_soup.find('table', id='senators')
tab_body = sen_table.find('tbody')
vcards = tab_body.find_all('span', class_='vcard')

spath = fpath + 'senators/'
if not os.path.exists(spath):
    os.mkdir(spath)

sen_urls = {}
for vcard in vcards:
    link = vcard.find('a')
    sen_urls[vcard.text] = seed_page + link.get('href')

for pol in sen_urls:
    pol_html = download_page(sen_urls[pol])
    with open(fpath + 'senators/' + pol + '.html', 'w') as outfile:
        outfile.write(pol_html)


#Extracting house data
reps_url = 'https://en.wikipedia.org/wiki/List_of_current_members_of_the_United_States_House_of_Representatives'
us_rep_html = download_page(reps_url)
us_rep_soup = BS(us_rep_html, 'html.parser')
rep_table = us_rep_soup.find('table', id='votingmembers')
tab_body = rep_table.find('tbody')
vcards = tab_body.find_all('b')

rpath = fpath + 'representatives/'
if not os.path.exists(rpath):
    os.mkdir(rpath)

for vcard in vcards:
    rlink = vcard.find('a')
    rep_path = fpath + 'representatives/' + str(unidecode.unidecode(vcard.text)) + '.html'
    with open(rep_path, 'w') as outfile:
        rhtml = download_page(seed_page + rlink.get('href'))
        outfile.write(rhtml)

files = os.listdir(rpath)
for file in files:
    if file.endswith('.html'):
        split_sents(rpath + file)

files = os.listdir(spath)
for file in files:
    if file.endswith('.html'):
        split_sents(spath + file)

