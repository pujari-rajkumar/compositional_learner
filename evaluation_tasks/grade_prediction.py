#!/usr/bin/env python
# coding: utf-8

#@author: Rajkumar Pujari

# In[1]:


import pickle
import numpy as np
import random
import math
from sklearn.metrics import confusion_matrix
import csv
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import matplotlib.pyplot as plt

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[5]:


fpath = './data/evaluation_data/'
log = open(fpath + 'grade_pred_log.txt', 'w')

# In[6]:


pol_ents = set()
with open(fpath + 'queries.pkl', 'rb') as infile:
    queries = pickle.load(infile)
    for query in queries:
        pol_ents.add(query[0][0])


# In[9]:


with open(fpath + 'entity_issue_bert.pkl', 'rb') as infile:
    ent_embs_bert = pickle.load(infile)
with open(fpath + 'entity_issue_bl.pkl', 'rb') as infile:
    ent_embs_bl = pickle.load(infile)
with open(fpath + 'entity_issue_encoder.pkl', 'rb') as infile:
    ent_embs_encoder = pickle.load(infile)
with open(fpath + 'entity_issue_model.pkl', 'rb') as infile:
    ent_embs_model = pickle.load(infile)


# In[10]:


class GradePredictor(nn.Module):
    
    def __init__(self, sizes, use_cuda, lr, momentum, weight_decay):
        super(GradePredictor, self).__init__()
        self.layers = []
        i = 0
        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            self.layers.append(nn.Linear(s1, s2))
            self.register_parameter('weight-layer-' + str(i), self.layers[-1].weight)
            self.register_parameter('bias-layer-' + str(i), self.layers[-1].bias)
            nn.init.xavier_uniform_(self.layers[-1].weight)
            i += 1
        self.loss_fn = nn.CrossEntropyLoss()
        self.nl = nn.Tanh()
        params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        
    def forward(self, x):
        for layer in self.layers:
            x = self.nl(layer(x))
        out = F.softmax(x, dim=1)
        return out
        
    def evaluate(self, data):
        self.eval()
        data_x, data_y = data
        pred_out = self.forward(data_x)
        pred_y = torch.argmax(pred_out, dim=1)
        acc = sum((pred_y == data_y).float()) / data_x.size(0)
        val_loss = self.loss_fn(pred_out, data_y)
        return acc, val_loss
        
    def train_model(self, train, val, num_epochs=10, save_path=fpath+'grade_predictor.pkl'):
        train_x, train_y = train
        train_out = self.forward(train_x)
        ids = list(range(train_x.size(0)))
        max_val = -1
        for i in range(num_epochs):
            self.train()
            random.shuffle(ids)
            train_x = train_x[ids, :]
            train_y = train_y[ids]
            train_out = self.forward(train_x)
            loss = self.loss_fn(train_out, train_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            val_acc, val_loss = self.evaluate(val)
            if val_acc > max_val:
                max_val = val_acc
                torch.save(self.state_dict(), save_path)


# # NRA Grade Experiments

# In[115]:


csv_file = csv.reader(open(fpath + 'nra-grades.csv', 'r'))
c = 0
grade_pol_ents = set()
rep_grades = set()
dem_grades = set()
nra_grades = {}
for row in csv_file:
    if row[5] in pol_ents:
        c += 1
        if row[5] not in nra_grades:
            nra_grades[row[5]] = []
        nra_grades[row[5]].append(row[-1])
print(c, len(grade_pol_ents), len(pol_ents), file=log, flush=True)


# In[116]:


pos_sent = 'I strongly support National Rifle Association (NRA)'
neg_sent = 'I vehemently oppose National Rifle Association (NRA)'


# In[117]:


with open(fpath + 'nra_sents.pkl', 'rb') as infile:
    nra_sent_embs = pickle.load(infile)


# In[118]:


res_file = open(fpath + 'nra_grade_res_all.txt', 'w')
random.seed(4056)
for ent in nra_grades:
    if ent in ent_embs_bert['guns']:
        print(ent, file=res_file, flush=True)
        n1_encoder_emb, n2_encoder_emb = ent_embs_encoder['guns'][ent]
        n1_bl_emb, n2_bl_emb = ent_embs_bl['guns'][ent]
        n1_bert_emb, n2_bert_emb = ent_embs_bert['guns'][ent]
        n1_model_emb, n2_model_emb = ent_embs_model['guns'][ent]
        
        e1_bert = (n1_bert_emb + n2_bert_emb).reshape(1, -1)
        cs_scores_bert = cos_sim(e1_bert, nra_sent_embs)
        
        e1_bl = (n1_bl_emb + n2_bl_emb).reshape(1, -1)
        cs_scores_bl = cos_sim(e1_bl, nra_sent_embs)
        
        e1_encoder = (n1_encoder_emb + n2_encoder_emb).reshape(1, -1)
        cs_scores_encoder = cos_sim(e1_encoder, nra_sent_embs)
        
        e1_model = (n1_model_emb + n2_model_emb).reshape(1, -1)
        cs_scores_model = cos_sim(e1_model, nra_sent_embs)
        
        print(np.argmax(cs_scores_bert.reshape(2,), axis=0), np.argmax(cs_scores_bl.reshape(2,), axis=0),              np.argmax(cs_scores_encoder.reshape(2,), axis=0), np.argmax(cs_scores_model.reshape(2,), axis=0),              nra_grades[ent], file=res_file, flush=True)
		
res_file.close()


# In[119]:


grade_2_idx = {
    'D+': 10,
    'A-': 2,
    'F': 13,
    'D': 11,
    'AQ': 3,
    'A': 1,
    'B+': 4,
    'B': 5,
    'C': 8,
    '?': -1,
    'C-': 9,
    'D-': 12,
    'C+': 7,
    'B-': 6,
    'A+': 0
}


# In[120]:


c_bert, c_bl, c_encoder, c_model, t, w = 0, 0, 0, 0, 0, 0
af_c_bert, af_c_bl, af_c_encoder, af_c_model, af_t, af_w = 0, 0, 0, 0, 0, 0
i = 0
grade_types = set()
f1 = open(fpath + 'nra_grade_res_all.txt', 'r')
for line in f1.readlines():
    if i % 2 == 0:
        ent_name = line.strip()
    elif i % 2 == 1:
        cols = line.replace('[', '').replace(']', '').replace(',', '').replace("'", '').strip().split()
        pred_affl_bert = int(cols[0])
        pred_affl_bl = int(cols[1])
        pred_affl_encoder = int(cols[2])
        pred_affl_model = int(cols[3])
        grade_nums = [grade_2_idx[g] for g in cols[4:]]
        true_affl_0 = [g <= 6.5 for g in grade_nums if g >=0]
        true_affl_1 = [g >= 6.5 for g in grade_nums if g >=0]
        af_true_affl_0 = [g <= 2 for g in grade_nums if g >=0]
        af_true_affl_1 = [g >= 13 for g in grade_nums if g >=0]
        
        if all(true_affl_0):
            true_affl = 0
        elif all(true_affl_1):
            true_affl = 1
        else:
            true_affl = -1
            
        if true_affl >= 0 and pred_affl_bert == true_affl:
            c_bert += 1
        if true_affl >= 0 and pred_affl_bl == true_affl:
            c_bl += 1
        if true_affl >= 0 and pred_affl_encoder == true_affl:
            c_encoder += 1
        if true_affl >= 0 and pred_affl_model == true_affl:
            c_model += 1
        
        if true_affl >= 0:
            t += 1
        else:
            w += 1
            
        if all(af_true_affl_0):
            af_true_affl = 0
        elif all(af_true_affl_1):
            af_true_affl = 1
        else:
            af_true_affl = -1
        
        if af_true_affl >= 0 and pred_affl_bert == af_true_affl:
            af_c_bert += 1
        if af_true_affl >= 0 and pred_affl_bl == af_true_affl:
            af_c_bl += 1
        if af_true_affl >= 0 and pred_affl_encoder == af_true_affl:
            af_c_encoder += 1
        if af_true_affl >= 0 and pred_affl_model == af_true_affl:
            af_c_model += 1
        
        if af_true_affl >= 0:
            af_t += 1
        else:
            af_w += 1
    i += 1
print('NRA Paraphrase Task Results', file=log, flush=True)
print('BERT:', c_bert, 'Adap:', c_bl, 'Enc:', c_encoder, 'Model:', c_model, 'Total:', t, file=log, flush=True)
print('A/F_BERT:', af_c_bert, 'A/F_Adap:', af_c_bl, 'A/F_Enc:', af_c_encoder, 'A/F_Model:', af_c_model, 'A/F_Total:', af_t, file=log, flush=True)


# In[121]:


def get_nra_label(ent):
    grade_nums = [gt_2_idx[g] for g in nra_grades[ent]]
    cd = {}
    ng = 0
    for g in grade_nums:
        if g >= 0:
            if g not in cd:
                cd[g] = 0
            cd[g] += 1
            ng += 1
    
    sorted_g = sorted([(k, v) for k, v in cd.items()], key=lambda x:x[1], reverse=True)
    if len(sorted_g) > 0 and sorted_g[0][1] >= ng-1:
        return sorted_g[0][0]
    else:
        return - 1


# In[122]:


gt_2_idx = {
    '?': -1,
    'F': 4,
    'D-': 3,
    'D': 3,
    'D+': 3,
    'C-': 2,
    'C': 2,
    'C+': 2,
    'B-': 1,
    'B': 1,
    'B+': 1,
    'AQ': 0,
    'A-': 0,
    'A': 0,
    'A+': 0
}


# In[123]:


def get_model_tensors(data):
    data_x = []
    data_y = []
    for eg in data:
        ent, embs, l = eg
        if l >= 0:
            inp_tensor = np.vstack([embs[0], embs[1]]).reshape(1, -1)
            data_x.append(inp_tensor)
            data_y.append(l)
    data_x = torch.from_numpy(np.vstack(data_x))
    data_y = torch.from_numpy(np.array(data_y))
    return data_x, data_y


# In[124]:

print('NRA Grade Prediction - BERT: ', file=log, flush=True)
data = []
for ent in ent_embs_bert['guns']:
    if ent in nra_grades:
        data.append((ent, ent_embs_bert['guns'][ent], get_nra_label(ent)))
data_x, data_y = get_model_tensors(data)
print(data_x.size(), data_y.size(), file=log, flush=True)
bert_test_res = {}
bert_val_res = {}
folds = 10
for num_tr in range(1, folds-1):
    bert_test_res[num_tr] = []
    bert_val_res[num_tr] = []
for seed in [5, 7, 11, 13, 17]:
    print('Seed,', seed, file=log, flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    data_ids = list(range(data_x.size(0)))
    random.shuffle(data_ids)
    bsz = int(data_x.size(0) / folds) + 1
    s = 0
    e = bsz
    batches = []
    while s < data_x.size(0):
        batches.append((data_x[s:e, :], data_y[s:e]))
        s = e
        e += bsz
    for num_tr in range(1, folds-1):
        test_accs = []
        val_accs = []
        for i in range(len(batches)):
            test = batches[i]
            val = batches[(i + 1)%folds]
            train = []
            j = (i + 2)%folds
            while j != i and len(train) < num_tr:
                train.append(batches[j])
                j = (j+1)%folds
            train_x = torch.cat([b[0] for b in train], dim=0) 
            train_y = torch.cat([b[1] for b in train], dim=0)
            gpred = GradePredictor([2*768, 1000, 5], False, 0.05, 0.4, 0)
            gpred.train_model((train_x, train_y), val)
            gpred.load_state_dict(torch.load(fpath + 'grade_predictor.pkl'))
            test_accs.append(gpred.evaluate(test)[0].unsqueeze(0))
            val_accs.append(gpred.evaluate(val)[0].unsqueeze(0))
        print(num_tr, ',', round(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100, 2), file=log, flush=True)
        bert_test_res[num_tr].append(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100)
        bert_val_res[num_tr].append(torch.mean(torch.cat(val_accs, dim=0)).numpy()*100)


# In[125]:

print('NRA Grade Prediction - BERT Adaptation: ', file=log, flush=True)
data = []
for ent in ent_embs_bl['guns']:
    if ent in nra_grades:
        data.append((ent, ent_embs_bl['guns'][ent], get_nra_label(ent)))
data_x, data_y = get_model_tensors(data)
print(data_x.size(), data_y.size(), file=log, flush=True)
bl_test_res = {}
bl_val_res = {}
folds = 10
for num_tr in range(1, folds-1):
    bl_test_res[num_tr] = []
    bl_val_res[num_tr] = []
for seed in [5, 7, 11, 13, 17]:
    print('Seed,', seed, file=log, flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    data_ids = list(range(data_x.size(0)))
    random.shuffle(data_ids)
    bsz = int(data_x.size(0) / folds) + 1
    s = 0
    e = bsz
    batches = []
    while s < data_x.size(0):
        batches.append((data_x[s:e, :], data_y[s:e]))
        s = e
        e += bsz
    for num_tr in range(1, folds-1):
        test_accs = []
        val_accs = []
        for i in range(len(batches)):
            test = batches[i]
            val = batches[(i + 1)%folds]
            train = []
            j = (i + 2)%folds
            while j != i and len(train) < num_tr:
                train.append(batches[j])
                j = (j+1)%folds
            train_x = torch.cat([b[0] for b in train], dim=0) 
            train_y = torch.cat([b[1] for b in train], dim=0)
            gpred = GradePredictor([2*768, 1000, 5], False, 0.05, 0.4, 0)
            gpred.train_model((train_x, train_y), val)
            gpred.load_state_dict(torch.load(fpath + 'grade_predictor.pkl'))
            test_accs.append(gpred.evaluate(test)[0].unsqueeze(0))
            val_accs.append(gpred.evaluate(val)[0].unsqueeze(0))
        print(num_tr, ',', round(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100, 2), file=log, flush=True)
        bl_test_res[num_tr].append(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100)
        bl_val_res[num_tr].append(torch.mean(torch.cat(val_accs, dim=0)).numpy()*100)


# In[126]:

print('NRA Grade Prediction - Encoder: ', file=log, flush=True)
data = []
for ent in ent_embs_encoder['guns']:
    if ent in nra_grades:
        data.append((ent, ent_embs_encoder['guns'][ent], get_nra_label(ent)))
data_x, data_y = get_model_tensors(data)
print(data_x.size(), data_y.size(), file=log, flush=True)
encoder_test_res = {}
encoder_val_res = {}
folds = 10
for num_tr in range(1, folds-1):
    encoder_test_res[num_tr] = []
    encoder_val_res[num_tr] = []
for seed in [5, 7, 11, 13, 17]:
    print('Seed,', seed, file=log, flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    data_ids = list(range(data_x.size(0)))
    random.shuffle(data_ids)
    bsz = int(data_x.size(0) / folds) + 1
    s = 0
    e = bsz
    batches = []
    while s < data_x.size(0):
        batches.append((data_x[s:e, :], data_y[s:e]))
        s = e
        e += bsz
    for num_tr in range(1, folds-1):
        test_accs = []
        val_accs = []
        for i in range(len(batches)):
            test = batches[i]
            val = batches[(i + 1)%folds]
            train = []
            j = (i + 2)%folds
            while j != i and len(train) < num_tr:
                train.append(batches[j])
                j = (j+1)%folds
            train_x = torch.cat([b[0] for b in train], dim=0) 
            train_y = torch.cat([b[1] for b in train], dim=0)
            gpred = GradePredictor([2*768, 1000, 5], False, 0.05, 0.4, 0)
            gpred.train_model((train_x, train_y), val)
            gpred.load_state_dict(torch.load(fpath + 'grade_predictor.pkl'))
            test_accs.append(gpred.evaluate(test)[0].unsqueeze(0))
            val_accs.append(gpred.evaluate(val)[0].unsqueeze(0))
        print(num_tr, ',', round(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100, 2), file=log, flush=True)
        encoder_test_res[num_tr].append(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100)
        encoder_val_res[num_tr].append(torch.mean(torch.cat(val_accs, dim=0)).numpy()*100)


# In[129]:

print('NRA Grade Prediction - Compositional Reader: ', file=log, flush=True)
data = []
for ent in ent_embs_model['guns']:
    if ent in nra_grades:
        data.append((ent, ent_embs_model['guns'][ent], get_nra_label(ent)))
data_x, data_y = get_model_tensors(data)
print(data_x.size(0), file=log, flush=True)
m_test_res = {}
m_val_res = {}
folds = 10
for num_tr in range(1, folds-1):
    m_test_res[num_tr] = []
    m_val_res[num_tr] = []
for seed in [5, 7, 11, 13, 17]:
    print('Seed,', seed, file=log, flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    data_ids = list(range(data_x.size(0)))
    random.shuffle(data_ids)
    bsz = int(data_x.size(0) / folds) + 1
    s = 0
    e = bsz
    batches = []
    while s < data_x.size(0):
        batches.append((data_x[s:e, :], data_y[s:e]))
        s = e
        e += bsz
    for num_tr in range(1, folds-1):
        test_accs = []
        val_accs = []
        for i in range(len(batches)):
            test = batches[i]
            val = batches[(i + 1)%folds]
            train = []
            j = (i + 2)%folds
            while j != i and len(train) < num_tr:
                train.append(batches[j])
                j = (j+1)%folds
            train_x = torch.cat([b[0] for b in train], dim=0) 
            train_y = torch.cat([b[1] for b in train], dim=0)
            gpred = GradePredictor([2*768, 1000, 5], False, 0.05, 0.4, 0)
            gpred.train_model((train_x, train_y), val)
            gpred.load_state_dict(torch.load(fpath + 'grade_predictor.pkl'))
            test_accs.append(gpred.evaluate(test)[0].unsqueeze(0))
            val_accs.append(gpred.evaluate(val)[0].unsqueeze(0))
        print(num_tr, ',', round(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100, 2), file=log, flush=True)
        m_test_res[num_tr].append(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100)
        m_val_res[num_tr].append(torch.mean(torch.cat(val_accs, dim=0)).numpy()*100)


# In[130]:

print('NRA Grade Prediction Validation Results: Num training folds, BERT_acc, BERT_std, BERT_Adap_acc, BERT_Adap_std, Encoder_acc, Encoder_std, Comp_Reader_acc, Comp_Reader_std', file=log, flush=True) 
for nr in bl_val_res:
    print(nr, ',', round(np.mean(bert_val_res[nr]), 2), ',', round(np.std(bert_val_res[nr]), 2),          ',', round(np.mean(bl_val_res[nr]), 2), ',', round(np.std(bl_val_res[nr]), 2),          ',', round(np.mean(encoder_val_res[nr]), 2), ',', round(np.std(encoder_val_res[nr]), 2),          ',', round(np.mean(m_val_res[nr]), 2), ',', round(np.std(m_val_res[nr]), 2), file=log, flush=True)


# In[131]:

print('NRA Grade Prediction Test Results: Num training folds, BERT_acc, BERT_std, BERT_Adap_acc, BERT_Adap_std, Encoder_acc, Encoder_std, Comp_Reader_acc, Comp_Reader_std', file=log, flush=True) 
for nr in bl_test_res:
    print(nr, ',', round(np.mean(bert_test_res[nr]), 2), ',', round(np.std(bert_test_res[nr]), 2),          ',', round(np.mean(bl_test_res[nr]), 2), ',', round(np.std(bl_test_res[nr]), 2),          ',', round(np.mean(encoder_test_res[nr]), 2), ',', round(np.std(encoder_test_res[nr]), 2),          ',', round(np.mean(m_test_res[nr]), 2), ',', round(np.std(m_test_res[nr]), 2), file=log, flush=True)


# In[159]:


plt.style.use('classic')

#Declare the array containing the series you want to plot. 
#For example:

n_steps = [ round(i*100/10., 2) for i in range(1, 9)]

bert_array = [round(np.mean(bert_test_res[nr]), 2) for nr in range(1,9)]
bl_array = [round(np.mean(bl_test_res[nr]), 2) for nr in range(1,9)]
m_array = [round(np.mean(m_test_res[nr]), 2) for nr in range(1,9)]


bert_std = [round(np.std(bert_test_res[nr]), 2) for nr in range(1,9)]
bl_std = [round(np.std(bl_test_res[nr]), 2) for nr in range(1,9)]
m_std = [round(np.std(m_test_res[nr]), 2) for nr in range(1,9)]

#Plotting:
plt.errorbar(n_steps, m_array, yerr=m_std, linewidth=2, markeredgewidth=1, ecolor='red',             marker='o', label='Compositional Reader')
plt.errorbar(n_steps, bl_array, yerr=bl_std, linewidth=2, markeredgewidth=1, ecolor='red',             color='olive', linestyle='--', marker='o', label='BERT Adaptation')
plt.errorbar(n_steps, bert_array, yerr=bert_std, linewidth=2, markeredgewidth=1, ecolor='red',             color='magenta', linestyle='-.', marker='o', label='BERT')

plt.legend()
plt.xlim(5, 85)
plt.ylim(25, 105)
plt.xlabel('Percentage Training Data Used (%)')
plt.ylabel('Test Accuracy (%)')
plt_name = 'nra_grade_pred_test_res.png'
plt.savefig(fpath + plt_name, dpi=150)
print(plt_name, file=log, flush=True)
#plt.show()
plt.close()


# In[133]:


plt.style.use('classic')

#Declare the array containing the series you want to plot. 
#For example:

n_steps = [ round(i*100/10., 2) for i in range(1, 9)]

bert_array = [round(np.mean(bert_val_res[nr]), 2) for nr in range(1,9)]
bl_array = [round(np.mean(bl_val_res[nr]), 2) for nr in range(1,9)]
m_array = [round(np.mean(m_val_res[nr]), 2) for nr in range(1,9)]


bert_std = [round(np.std(bert_val_res[nr]), 2) for nr in range(1,9)]
bl_std = [round(np.std(bl_val_res[nr]), 2) for nr in range(1,9)]
m_std = [round(np.std(m_val_res[nr]), 2) for nr in range(1,9)]

#Plotting:
plt.errorbar(n_steps, m_array, yerr=m_std, linewidth=2, markeredgewidth=1, ecolor='red',             marker='o', label='Compositional Reader')
plt.errorbar(n_steps, bl_array, yerr=bl_std, linewidth=2, markeredgewidth=1, ecolor='red',             color='olive', linestyle='--', marker='o', label='BERT Adaptation')
plt.errorbar(n_steps, bert_array, yerr=bert_std, linewidth=2, markeredgewidth=1, ecolor='red',             color='magenta', linestyle='-.', marker='o', label='BERT')

plt.legend()
plt.xlim(5, 85)
plt.ylim(25, 110)
plt.xlabel('Percentage Training Data Used (%)')
plt.ylabel('Validation Accuracy (%)')
plt_name = 'nra_grade_pred_val_res.png'
plt.savefig(fpath + plt_name, dpi=150)
print(plt_name, file=log, flush=True)
#plt.show()
plt.close()

# # LCV Score Experiments

# In[190]:


csv_file = csv.reader(open(fpath + '2019_lcv_scores.csv', 'r'))
c = 0
grade_pol_ents = set()
lcv_scores = {}
for row in csv_file:
    cols = [col.strip() for col in row[2].split(',')]
    pol_name = ' '.join(cols[1:] + [cols[0]]) 
#     print(pol_name)
    if pol_name in pol_ents:
        c += 1
        if pol_name not in lcv_scores:
            lcv_scores[pol_name] = []
        lcv_scores[pol_name].append(int(row[4]))
print(c, len(grade_pol_ents), len(pol_ents), file=log, flush=True)


# In[194]:


num_classes = 4
def get_lcv_label(ent):
    score = lcv_scores[ent][0]
    if score <= 25:
        return 0
    elif score <= 50:
        return 1
    elif score <= 75:
        return 2
    elif score <= 100:
        return 3


# In[195]:

print('LCV Score Prediction - BERT: ', file=log, flush=True)
data = []
for ent in ent_embs_bert['environment']:
    if ent in lcv_scores:
        data.append((ent, ent_embs_bert['environment'][ent], get_lcv_label(ent)))
data_x, data_y = get_model_tensors(data)
print(data_x.size(), data_y.size(), file=log, flush=True)
bert_test_res = {}
bert_val_res = {}
folds = 10
for num_tr in range(1, folds-1):
    bert_test_res[num_tr] = []
    bert_val_res[num_tr] = []
for seed in [5, 7, 11, 13, 17]:
    print('Seed,', seed, file=log, flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    data_ids = list(range(data_x.size(0)))
    random.shuffle(data_ids)
    bsz = int(data_x.size(0) / folds) + 1
    s = 0
    e = bsz
    batches = []
    while s < data_x.size(0):
        batches.append((data_x[s:e, :], data_y[s:e]))
        s = e
        e += bsz
    for num_tr in range(1, folds-1):
        test_accs = []
        val_accs = []
        for i in range(len(batches)):
            test = batches[i]
            val = batches[(i + 1)%folds]
            train = []
            j = (i + 2)%folds
            while j != i and len(train) < num_tr:
                train.append(batches[j])
                j = (j+1)%folds
            train_x = torch.cat([b[0] for b in train], dim=0) 
            train_y = torch.cat([b[1] for b in train], dim=0)
            gpred = GradePredictor([2*768, 1000, num_classes], False, 0.05, 0.4, 0)
            gpred.train_model((train_x, train_y), val)
            gpred.load_state_dict(torch.load(fpath + 'grade_predictor.pkl'))
            test_accs.append(gpred.evaluate(test)[0].unsqueeze(0))
            val_accs.append(gpred.evaluate(val)[0].unsqueeze(0))
        print(num_tr, ',', round(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100, 2), file=log, flush=True)
        bert_test_res[num_tr].append(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100)
        bert_val_res[num_tr].append(torch.mean(torch.cat(val_accs, dim=0)).numpy()*100)


# In[196]:

print('LCV Score Prediction - BERT Adaptation: ', file=log, flush=True)
data = []
for ent in ent_embs_bl['environment']:
    if ent in lcv_scores:
        data.append((ent, ent_embs_bl['environment'][ent], get_lcv_label(ent)))
data_x, data_y = get_model_tensors(data)
print(data_x.size(), data_y.size(), file=log, flush=True)
bl_test_res = {}
bl_val_res = {}
folds = 10
for num_tr in range(1, folds-1):
    bl_test_res[num_tr] = []
    bl_val_res[num_tr] = []
for seed in [5, 7, 11, 13, 17]:
    print('Seed,', seed, file=log, flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    data_ids = list(range(data_x.size(0)))
    random.shuffle(data_ids)
    bsz = int(data_x.size(0) / folds) + 1
    s = 0
    e = bsz
    batches = []
    while s < data_x.size(0):
        batches.append((data_x[s:e, :], data_y[s:e]))
        s = e
        e += bsz
    for num_tr in range(1, folds-1):
        test_accs = []
        val_accs = []
        for i in range(len(batches)):
            test = batches[i]
            val = batches[(i + 1)%folds]
            train = []
            j = (i + 2)%folds
            while j != i and len(train) < num_tr:
                train.append(batches[j])
                j = (j+1)%folds
            train_x = torch.cat([b[0] for b in train], dim=0) 
            train_y = torch.cat([b[1] for b in train], dim=0)
            gpred = GradePredictor([2*768, 1000, num_classes], False, 0.05, 0.4, 0)
            gpred.train_model((train_x, train_y), val)
            gpred.load_state_dict(torch.load(fpath + 'grade_predictor.pkl'))
            test_accs.append(gpred.evaluate(test)[0].unsqueeze(0))
            val_accs.append(gpred.evaluate(val)[0].unsqueeze(0))
        print(num_tr, ',', round(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100, 2), file=log, flush=True)
        bl_test_res[num_tr].append(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100)
        bl_val_res[num_tr].append(torch.mean(torch.cat(val_accs, dim=0)).numpy()*100)


# In[197]:

print('LCV Score Prediction - Encoder: ', file=log, flush=True)
data = []
for ent in ent_embs_encoder['environment']:
    if ent in lcv_scores:
        data.append((ent, ent_embs_encoder['environment'][ent], get_lcv_label(ent)))
data_x, data_y = get_model_tensors(data)
print(data_x.size(), data_y.size(), file=log, flush=True)
encoder_test_res = {}
encoder_val_res = {}
folds = 10
for num_tr in range(1, folds-1):
    encoder_test_res[num_tr] = []
    encoder_val_res[num_tr] = []
for seed in [5, 7, 11, 13, 17]:
    print('Seed,', seed, file=log, flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    data_ids = list(range(data_x.size(0)))
    random.shuffle(data_ids)
    bsz = int(data_x.size(0) / folds) + 1
    s = 0
    e = bsz
    batches = []
    while s < data_x.size(0):
        batches.append((data_x[s:e, :], data_y[s:e]))
        s = e
        e += bsz
    for num_tr in range(1, folds-1):
        test_accs = []
        val_accs = []
        for i in range(len(batches)):
            test = batches[i]
            val = batches[(i + 1)%folds]
            train = []
            j = (i + 2)%folds
            while j != i and len(train) < num_tr:
                train.append(batches[j])
                j = (j+1)%folds
            train_x = torch.cat([b[0] for b in train], dim=0) 
            train_y = torch.cat([b[1] for b in train], dim=0)
            gpred = GradePredictor([2*768, 1000, num_classes], False, 0.05, 0.4, 0)
            gpred.train_model((train_x, train_y), val)
            gpred.load_state_dict(torch.load(fpath + 'grade_predictor.pkl'))
            test_accs.append(gpred.evaluate(test)[0].unsqueeze(0))
            val_accs.append(gpred.evaluate(val)[0].unsqueeze(0))
        print(num_tr, ',', round(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100, 2), file=log, flush=True)
        encoder_test_res[num_tr].append(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100)
        encoder_val_res[num_tr].append(torch.mean(torch.cat(val_accs, dim=0)).numpy()*100)


# In[198]:

print('LCV Score Prediction - Compositional Reader: ', file=log, flush=True)
data = []
for ent in ent_embs_model['environment']:
    if ent in lcv_scores:
        data.append((ent, ent_embs_model['environment'][ent], get_lcv_label(ent)))
data_x, data_y = get_model_tensors(data)
print(data_x.size(), data_y.size(), file=log, flush=True)
m_test_res = {}
m_val_res = {}
folds = 10
for num_tr in range(1, folds-1):
    m_test_res[num_tr] = []
    m_val_res[num_tr] = []
for seed in [5, 7, 11, 13, 17]:
    print('Seed,', seed, file=log, flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    data_ids = list(range(data_x.size(0)))
    random.shuffle(data_ids)
    bsz = int(data_x.size(0) / folds) + 1
    s = 0
    e = bsz
    batches = []
    while s < data_x.size(0):
        batches.append((data_x[s:e, :], data_y[s:e]))
        s = e
        e += bsz
    for num_tr in range(1, folds-1):
        test_accs = []
        val_accs = []
        for i in range(len(batches)):
            test = batches[i]
            val = batches[(i + 1)%folds]
            train = []
            j = (i + 2)%folds
            while j != i and len(train) < num_tr:
                train.append(batches[j])
                j = (j+1)%folds
            train_x = torch.cat([b[0] for b in train], dim=0) 
            train_y = torch.cat([b[1] for b in train], dim=0)
            gpred = GradePredictor([2*768, 1000, num_classes], False, 0.05, 0.4, 0)
            gpred.train_model((train_x, train_y), val)
            gpred.load_state_dict(torch.load(fpath + 'grade_predictor.pkl'))
            test_accs.append(gpred.evaluate(test)[0].unsqueeze(0))
            val_accs.append(gpred.evaluate(val)[0].unsqueeze(0))
        print(num_tr, ',', round(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100, 2), file=log, flush=True)
        m_test_res[num_tr].append(torch.mean(torch.cat(test_accs, dim=0)).numpy()*100)
        m_val_res[num_tr].append(torch.mean(torch.cat(val_accs, dim=0)).numpy()*100)


# In[199]:

print('LCV Score Prediction Test Results: Num training folds, BERT_acc, BERT_std, BERT_Adap_acc, BERT_Adap_std, Encoder_acc, Encoder_std, Comp_Reader_acc, Comp_Reader_std', file=log, flush=True) 
for nr in bl_test_res:
    print(nr, ',', round(np.mean(bert_test_res[nr]), 2), ',', round(np.std(bert_test_res[nr]), 2),          ',', round(np.mean(bl_test_res[nr]), 2), ',', round(np.std(bl_test_res[nr]), 2),          ',', round(np.mean(encoder_test_res[nr]), 2), ',', round(np.std(encoder_test_res[nr]), 2),          ',', round(np.mean(m_test_res[nr]), 2), ',', round(np.std(m_test_res[nr]), 2), file=log, flush=True)


# In[200]:

print('LCV Score Prediction Validation Results: Num training folds, BERT_acc, BERT_std, BERT_Adap_acc, BERT_Adap_std, Encoder_acc, Encoder_std, Comp_Reader_acc, Comp_Reader_std', file=log, flush=True) 
for nr in bl_val_res:
    print(nr, ',', round(np.mean(bert_val_res[nr]), 2), ',', round(np.std(bert_val_res[nr]), 2),          ',', round(np.mean(bl_val_res[nr]), 2), ',', round(np.std(bl_val_res[nr]), 2),          ',', round(np.mean(encoder_val_res[nr]), 2), ',', round(np.std(encoder_val_res[nr]), 2),          ',', round(np.mean(m_val_res[nr]), 2), ',', round(np.std(m_val_res[nr]), 2), file=log, flush=True)


# In[201]:


plt.style.use('classic')

#Declare the array containing the series you want to plot. 
#For example:

n_steps = [ round(i*100/10., 2) for i in range(1, 9)]

bert_array = [round(np.mean(bert_test_res[nr]), 2) for nr in range(1,9)]
bl_array = [round(np.mean(bl_test_res[nr]), 2) for nr in range(1,9)]
m_array = [round(np.mean(m_test_res[nr]), 2) for nr in range(1,9)]


bert_std = [round(np.std(bert_test_res[nr]), 2) for nr in range(1,9)]
bl_std = [round(np.std(bl_test_res[nr]), 2) for nr in range(1,9)]
m_std = [round(np.std(m_test_res[nr]), 2) for nr in range(1,9)]

#Plotting:
plt.errorbar(n_steps, m_array, yerr=m_std, linewidth=2, markeredgewidth=1, ecolor='red',             marker='o', label='Compositional Reader')
plt.errorbar(n_steps, bl_array, yerr=bl_std, linewidth=2, markeredgewidth=1, ecolor='red',             color='olive', linestyle='--', marker='o', label='BERT Adaptation')
plt.errorbar(n_steps, bert_array, yerr=bert_std, linewidth=2, markeredgewidth=1, ecolor='red',             color='magenta', linestyle='-.', marker='o', label='BERT')

plt.legend()
plt.xlim(5, 85)
plt.ylim(40, 75)
plt.xlabel('Percentage Training Data Used (%)')
plt.ylabel('Test Accuracy (%)')
plt_name = 'lcv_score_pred_test_res.png'
plt.savefig(fpath + plt_name, dpi=150)
print(plt_name, file=log, flush=True)
#plt.show()
plt.close()

# In[202]:


plt.style.use('classic')

#Declare the array containing the series you want to plot. 
#For example:

n_steps = [ round(i*100/10., 2) for i in range(1, 9)]

bert_array = [round(np.mean(bert_val_res[nr]), 2) for nr in range(1,9)]
bl_array = [round(np.mean(bl_val_res[nr]), 2) for nr in range(1,9)]
m_array = [round(np.mean(m_val_res[nr]), 2) for nr in range(1,9)]


bert_std = [round(np.std(bert_val_res[nr]), 2) for nr in range(1,9)]
bl_std = [round(np.std(bl_val_res[nr]), 2) for nr in range(1,9)]
m_std = [round(np.std(m_val_res[nr]), 2) for nr in range(1,9)]

#Plotting:
plt.errorbar(n_steps, m_array, yerr=m_std, linewidth=2, markeredgewidth=1, ecolor='red',             marker='o', label='Compositional Reader')
plt.errorbar(n_steps, bl_array, yerr=bl_std, linewidth=2, markeredgewidth=1, ecolor='red',             color='olive', linestyle='--', marker='o', label='BERT Adaptation')
plt.errorbar(n_steps, bert_array, yerr=bert_std, linewidth=2, markeredgewidth=1, ecolor='red',             color='magenta', linestyle='-.', marker='o', label='BERT')

plt.legend()
plt.xlim(5, 85)
plt.ylim(40, )
plt.xlabel('Percentage Training Data Used (%)')
plt.ylabel('Validation Accuracy (%)')
plt_name = 'lcv_score_pred_val_res.png'
plt.savefig(fpath + plt_name, dpi=150)
print(plt_name, file=log, flush=True)
#plt.show()
plt.close()


log.close()
