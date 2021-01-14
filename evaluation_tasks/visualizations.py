#!/usr/bin/env python
# coding: utf-8

# @author: Rajkumar Pujari


import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
import csv
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import matplotlib.pyplot as plt


fpath = './data/evaluation_data/'


pol_ents = set()
issues = set()
with open(fpath + 'queries.pkl', 'rb') as infile:
    queries = pickle.load(infile)
    for query in queries:
        pol_ents.add(query[0][0])
        issues.add(query[1][0])
print(issues)



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
            grade_pol_ents.add(row[5])
        nra_grades[row[5]].append(row[-1])
print(c, len(grade_pol_ents), len(pol_ents))



with open(fpath + 'entity_issue_bert.pkl', 'rb') as infile:
    ent_embs_bert = pickle.load(infile)
with open(fpath + 'entity_issue_bl.pkl', 'rb') as infile:
    ent_embs_bl = pickle.load(infile)
with open(fpath + 'entity_issue_encoder.pkl', 'rb') as infile:
    ent_embs_encoder = pickle.load(infile)
with open(fpath + 'entity_issue_model.pkl', 'rb') as infile:
    ent_embs_model = pickle.load(infile)



from sklearn.decomposition import PCA
pca = PCA(n_components=2)


csv_file = csv.reader(open(fpath + 'nra-grades.csv', 'r'))
i = 0
ent_affl = {}
for row in csv_file:
    if i == 0:
        pass
    else:
        ent_affl[row[5]] = row[6]
    i += 1



plt.style.use('classic')
for issue in issues:
    print(issue)
    data_labels = []
    data_coords = []
    for ent in ent_embs_model[issue]:
        if ent in ent_affl:
            data_labels.append(ent)
            n1_emb, n2_emb = ent_embs_model[issue][ent]
            data_coords.append((n1_emb + n2_emb))
    data_coords = np.vstack(data_coords)
    print(data_coords.shape)
    out = pca.fit_transform(data_coords)
    colors = []
    labels = []
    rep_x, rep_y = [], []
    dem_x, dem_y = [], []
    oth_x, oth_y = [], []
    for i, ent in enumerate(data_labels):
        if ent_affl[ent] == 'R':
            if out[i, 0] < 10:
                rep_x.append(out[i, 0])
                rep_y.append(out[i, 1])
        elif ent_affl[ent] == 'D':
            if out[i, 0] < 10:
                dem_x.append(out[i, 0])
                dem_y.append(out[i, 1])
        else:
            if out[i, 0] < 10:
                oth_x.append(out[i, 0])
                oth_y.append(out[i, 1])
    plt.scatter(rep_x, rep_y, c='tab:red', marker='o', label='Republican', edgecolors='none')
    plt.scatter(dem_x, dem_y, c='b', marker='^', label='Democrat', edgecolors='none')
    plt.scatter(oth_x, oth_y, c='tab:green', marker='+', label='Other', edgecolors='none')
    plt_name = 'reps_dems_' + issue + '.png'
    plt.savefig(fpath + plt_name, dpi=150)
    print(plt_name)
    plt.legend()
    plt.show()
    plt.close()



#Entities with represntation for all issues and also have NRA grades
ent_set = set(ent_affl.keys())
for issue in issues:
    il = list(ent_embs_model[issue].keys())
    il_set = set(il)
    ent_set = ent_set & il_set


ent_pairs = [
    ['Chris Coons', 'Mitch McConnell', 'Kamala Harris'],
    ['Francis Rooney', 'Elizabeth Warren', 'Bernie Sanders'],
    ['Steve Womack', 'Mitch McConnell', 'Nancy Pelosi'],
    ['Mitch McConnell', 'Francis Rooney', 'Bernie Sanders']
]
        
issue_pairs = [
    ['guns', 'gay-rights'],
    ['environment', 'gay-rights'],
    ['guns', 'environment'],
    ['guns', 'environment']
]

for ent_pair, sel_issues in zip(ent_pairs, issue_pairs):
    data_coords = []
    data_labels = []
    colors = []
    for issue in sel_issues:
        for i, ent in enumerate(ent_pair):
            n1_emb, n2_emb = ent_embs_model[issue][ent]
            data_coords.append((n1_emb + n2_emb).reshape(1, -1))
            data_labels.append(ent + ' - ' + issue)
            if i == 0:
                colors.append('r')
            if i == 1:
                colors.append('g')
            if i == 2:
                colors.append('b')
    data_coords = np.vstack(data_coords)
    out = pca.fit_transform(data_coords)
    i1_x = []
    i1_y = []

    i2_x = []
    i2_y = []

    i3_x = []
    i3_y = []

    i1_c = []
    i2_c = []
    i3_c = []

    i1_l = []
    i2_l = []
    i3_l = []

    for i in range(out.shape[0]):
        if i < 2:
            i1_x.append(out[i, 0])
            i1_y.append(out[i, 1])
            i1_c.append(colors[i])
            i1_l.append(data_labels[i])
        elif i < 4:
            i2_x.append(out[i, 0])
            i2_y.append(out[i, 1])
            i2_c.append(colors[i])
            i2_l.append(data_labels[i])
        else:
            i3_x.append(out[i, 0])
            i3_y.append(out[i, 1])
            i3_c.append(colors[i])
            i3_l.append(data_labels[i])
    plt.scatter(i1_x, i1_y, c=i1_c, marker='o', edgecolors='none')
    x_offset = 1
    y_offset = -0.1
    plt.annotate(i1_l[0], (i1_x[0] + x_offset, i1_y[0] + y_offset))
    plt.annotate(i1_l[1], (i1_x[1] + x_offset, i1_y[1] + y_offset))
    plt.scatter(i2_x, i2_y, c=i2_c, marker='x', edgecolors='none')
    plt.annotate(i2_l[0], (i2_x[0] + x_offset, i2_y[0] + y_offset))
    plt.annotate(i2_l[1], (i2_x[1] + x_offset, i2_y[1] + y_offset))
    plt.scatter(i3_x, i3_y, c=i3_c, marker='^', edgecolors='none')
    plt.annotate(i3_l[0], (i3_x[0] + x_offset, i3_y[0] + y_offset))
    plt.annotate(i3_l[1], (i3_x[1] + x_offset, i3_y[1] + y_offset))
    plt.xlim(-25, 50)
    plt_name = '_'.join([ent.lower().split()[-1] for ent in ent_pair] + sel_issues) + '.png'
    plt.savefig(fpath + plt_name, dpi=150)
    print(plt_name)
    plt.show()
    plt.close()
