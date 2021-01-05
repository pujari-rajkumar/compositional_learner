#!/usr/bin/env python
# coding: utf-8

#@author: Rajkumar Pujari

# In[3]:


import pickle


# In[4]:


with open('./data/composite_learner_data/wiki_ann_dict.pkl', 'rb') as infile:
    wiki_ann_dict = pickle.load(infile)


# In[5]:


def get_mentioned_ent_set(data_dict):
    ret_ents = {}
    
    if 'news' in data_dict:
        news = data_dict['news'][1]
        for topic in news:
            for event in news[topic]:
                for doc in news[topic][event]:
                    for sent in doc:
                        for ment_ent in sent:
                            if ment_ent not in ret_ents:
                                ret_ents[ment_ent] = 0
                            ret_ents[ment_ent] += 1
    
    if 'description' in data_dict:
        description = data_dict['description'][1]
        for topic in description:
            for doc in description[topic]:
                for sent in doc:
                    for ment_ent in sent:
                        if ment_ent not in ret_ents:
                            ret_ents[ment_ent] = 0
                        ret_ents[ment_ent] += 1

    for ent in data_dict:
        if ent not in ['news', 'description']:
            if 'tweets' in data_dict[ent]:
                tweets = data_dict[ent]['tweets'][1]
                for topic in tweets:
                    for event in tweets[topic]:
                        for doc in tweets[topic][event]:
                            for sent in doc:
                                for ment_ent in sent:
                                    if ment_ent not in ret_ents:
                                        ret_ents[ment_ent] = 0
                                    ret_ents[ment_ent] += 1
            
            if 'statements' in data_dict[ent]:
                statements = data_dict[ent]['statements'][1]
                for topic in statements:
                    for event in statements[topic]:
                        for doc in statements[topic][event]:
                            for sent in doc:
                                for ment_ent in sent:
                                    if ment_ent not in ret_ents:
                                        ret_ents[ment_ent] = 0
                                    ret_ents[ment_ent] += 1
            
            if 'quotes' in data_dict[ent]:
                quotes = data_dict[ent]['quotes'][1]
                for topic in quotes:
                    for doc in quotes[topic]:
                        for sent in doc:
                            for ment_ent in sent:
                                if ment_ent not in ret_ents:
                                    ret_ents[ment_ent] = 0
                                ret_ents[ment_ent] += 1
            
            if 'wiki' in data_dict[ent]:
                wiki = data_dict[ent]['wiki'][1]
                for doc in wiki:
                    for sent in doc:
                        for ment_ent in sent:
                            if ment_ent not in ret_ents:
                                ret_ents[ment_ent] = 0
                            ret_ents[ment_ent] += 1
    
    return ret_ents


# In[6]:


def wikify_mentioned_ents(ent_set):
    link_to_name = {}
    for ent in ent_set:
        freq = ent_set[ent]
        cand_url = wiki_ann_dict[ent][0]['url']
        if cand_url in link_to_name:
            elist, efreq = link_to_name[cand_url]
            elist.append(ent)
            link_to_name[cand_url] = (elist, efreq + freq)
        else:
            link_to_name[cand_url] = ([ent], freq)
    return link_to_name


# In[7]:

def get_wiki_name(ment):
    cand_url = wiki_ann_dict[ment][0]['url']
    return cand_url.split('/')[-1]


def add_node(n_name, nodes, node_idx, adj_list):
    if n_name not in nodes:
        nodes[n_name] = node_idx
        adj_list[node_idx] = []
        node_idx += 1
    return nodes, adj_list, node_idx

def add_edge(nodes, node_idx, adj_list, src_name, dest_name):
    if src_name not in nodes:
        nodes, adj_list, node_idx = add_node(src_name, nodes, node_idx, adj_list)
    if dest_name not in nodes:
        nodes, adj_list, node_idx = add_node(dest_name, nodes, node_idx, adj_list)
    if nodes[dest_name] not in adj_list[nodes[src_name]]:
        adj_list[nodes[src_name]].append(nodes[dest_name])
    return nodes, adj_list, node_idx


# In[8]:


def add_doc(nodes, adj_list, node_idx, doc, ent, src_name, dname, wlinks, threshold, bidirectional=False):
    nodes, adj_list, node_idx = add_node(dname, nodes, node_idx, adj_list)
    nodes, adj_list, node_idx = add_edge(nodes, node_idx, adj_list, src_name, dname)
    if bidirectional:
        nodes, adj_list, node_idx = add_edge(nodes, node_idx, adj_list, dname, src_name)
    for sent in ent:
        for ment in sent:
            if wlinks[wiki_ann_dict[ment][0]['url']][1] >= threshold:
                ename = 'ref-' + wiki_ann_dict[ment][0]['url'].split('/')[-1]
                if ename not in nodes:
                    nodes, adj_list, node_idx = add_node(ename, nodes, node_idx, adj_list)
                nodes, adj_list, node_idx = add_edge(nodes, node_idx, adj_list, ename, dname)
    return nodes, adj_list, node_idx


# In[14]:


def generate_graph(data_dict, threshold=40):
    nodes = {}
    adj_list = {}
    node_idx = 0
    
    #Create mentioned entity nodes
    ment_entities = get_mentioned_ent_set(data_dict)
    wlinks = wikify_mentioned_ents(ment_entities)
    for ment in wlinks:
        if wlinks[ment][1] >= threshold:
            ename = 'ref-' + ment.split('/')[-1]
            nodes, adj_list, node_idx = add_node(ename, nodes, node_idx, adj_list)
    #Creating entity nodes
    for ent in data_dict:
        if ent not in ['description', 'news']:
            ename = 'author-' + ent
            nodes, adj_list, node_idx = add_node(ename, nodes, node_idx, adj_list)
    #Creating issue and event nodes
    if 'news' in data_dict:
        for topic in data_dict['news'][0]:
            iname = 'issue-' + topic
            nodes, adj_list, node_idx = add_node(iname, nodes, node_idx, adj_list)
            for event in data_dict['news'][0][topic]:
                en_name = 'event-' + topic + '-' + str(event)
                nodes, adj_list, node_idx = add_node(en_name, nodes, node_idx, adj_list)
                nodes, adj_list, node_idx = add_edge(nodes, node_idx, adj_list, iname, en_name)
                nodes, adj_list, node_idx = add_edge(nodes, node_idx, adj_list, en_name, iname)
    
    #Creating description document nodes
    if 'description' in data_dict:
        docs, ents, _ = data_dict['description']
        for topic in data_dict['description'][0]:
            i = 0
            for doc, ent in zip(docs[topic], ents[topic]):
                dname = 'doc-description-' + topic + '-' + str(i)
                nodes, adj_list, node_idx = add_doc(nodes, adj_list, node_idx, doc, ent, 'issue-' + topic, dname, wlinks, threshold)
                i += 1
    
    #Creating news document nodes
    if 'news' in data_dict:
        docs, ents, _ = data_dict['news']
        for topic in data_dict['news'][0]:
            for event in data_dict['news'][0][topic]:
                i = 0
                src_name = 'event-' + topic + '-' + str(event)
                for doc, ent in zip(docs[topic][event], ents[topic][event]):
                    dname = 'doc-news-' + topic + '-' + str(event) + '-' + str(i)
                    nodes, adj_list, node_idx = add_doc(nodes, adj_list, node_idx, doc, ent, src_name, dname, wlinks, threshold, bidirectional=True)
                    i += 1
                
    for author in data_dict:       
        if author in ['news', 'description']:
            continue
        #Creating wiki document nodes
        i = 0
        if 'wiki' in data_dict[author]:
            docs, ents, _ = data_dict[author]['wiki']
            for doc, ent in zip(docs, ents):
                dname = 'doc-wiki-' + author + '-' + str(i)
                nodes, adj_list, node_idx = add_doc(nodes, adj_list, node_idx, doc, ent, 'author-' + author, dname, wlinks, threshold)
                i += 1
        #Creating quote document nodes
        if 'quotes' in data_dict[author]:
            docs, ents, _ = data_dict[author]['quotes']
            for topic in docs:
                for doc, ent in zip(docs[topic], ents[topic]):
                    dname = 'doc-quote-' + author + '-' + topic
                    nodes, adj_list, node_idx = add_doc(nodes, adj_list, node_idx, doc, ent, 'author-' + author, dname, wlinks, threshold, bidirectional=True)
                    nodes, adj_list, node_idx = add_edge(nodes, node_idx, adj_list, 'issue-' + topic, dname)
        
        #Creating tweet document nodes
        if 'tweets' in data_dict[author]:
            docs, ents, _ = data_dict[author]['tweets']
            for topic in docs:
                for event in docs[topic]:
                    i = 0
                    for doc, ent in zip(docs[topic][event], ents[topic][event]):
                        dname = 'doc-tweet-' + author + '-' + topic + '-' + str(event) + '-' + str(i)
                        nodes, adj_list, node_idx = add_doc(nodes, adj_list, node_idx, doc, ent, 'author-' + author, dname, wlinks, threshold, bidirectional=True)
                        nodes, adj_list, node_idx = add_edge(nodes, node_idx, adj_list, 'event-' + topic + '-' + str(event), dname)
                        i += 1
        #Creating statement document nodes
        if 'statements' in data_dict[author]:
            docs, ents, _ = data_dict[author]['statements']
            for topic in docs:
                for event in docs[topic]:
                    i = 0
                    for doc, ent in zip(docs[topic][event], ents[topic][event]):
                        dname = 'doc-statement-' + author + '-' + topic + '-' + str(event) + '-' + str(i)
                        nodes, adj_list, node_idx = add_doc(nodes, adj_list, node_idx, doc, ent, 'author-' + author, dname, wlinks, threshold, bidirectional=True)
                        nodes, adj_list, node_idx = add_edge(nodes, node_idx, adj_list, 'event-' + topic + '-' + str(event), dname)
                        i += 1
    
    idx_to_names = {}
    for node_name in nodes:
        node_idx = nodes[node_name]
        idx_to_names[node_idx] = node_name
    
    return nodes, idx_to_names, adj_list

