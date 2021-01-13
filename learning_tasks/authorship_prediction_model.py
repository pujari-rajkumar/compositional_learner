#!/usr/bin/env python
# coding: utf-8

#@author: Rajkumar Pujari

import pickle
import numpy as np
from datetime import datetime
import copy
import random
import math
from sklearn.metrics import confusion_matrix
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from query_handling import data_splitting_utils
from query_handling import graph_generation


from compositional_learner.encoder import Encoder
from compositional_learner.composer import Composer


from learning_tasks.authorship_prediction_utils import load_data_batch, batchify_data


fpath = './data/composite_learner_data/'


class LinkPredictor(nn.Module):
    
    def __init__(self, args):
        super(LinkPredictor, self).__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.composer = Composer(args)
        self.ff1 = nn.Linear(5 * args.d_encoder, args.d_hidden)
        self.nl1 = nn.Tanh()
        self.do1 = nn.Dropout(p=args.dropout)
        self.ff2 = nn.Linear(args.d_hidden, args.n_classes)
        self.nl2 = nn.Tanh()
        self.loss_fn = nn.CrossEntropyLoss()
        params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(params, lr=args.lr, momentum=0.4, weight_decay=args.weight_decay)
        if args.pretrained:
            self.load_state_dict(torch.load(args.pretrained))
        if args.init_linear:
            nn.init.xavier_uniform_(self.ff1.weight)
            nn.init.xavier_uniform_(self.ff2.weight)
            
        
    def forward(self, node_docs, node_mask, comb_amtx, src_ids, dest_ids):
        batch_size = node_docs.size(0)
        if self.args.cuda:
            node_docs = node_docs.cuda()
            node_mask = node_mask.cuda()
            comb_amtx = comb_amtx.cuda()
            src_ids = src_ids.cuda()
            dest_ids = dest_ids.cuda()
        graph_node_embs = self.encoder(node_docs, node_mask)
        src_init_embs = graph_node_embs[torch.arange(batch_size), src_ids]
        dest_init_embs = graph_node_embs[torch.arange(batch_size), dest_ids]
        updated_node_embs, summary_emb = self.composer(graph_node_embs, comb_amtx)
        src_embs = updated_node_embs[-1][torch.arange(batch_size), src_ids]
        dest_embs  = updated_node_embs[-1][torch.arange(batch_size), dest_ids]
        ff_input = torch.cat([src_init_embs, dest_init_embs, src_embs, dest_embs, summary_emb], dim=-1)
        ff1_l = self.ff1(ff_input)
        ff1_nl = self.nl1(ff1_l)
        ff1_out = self.do1(ff1_nl)
        ff2_l = self.ff2(ff1_out)
        ff2_out = self.nl2(ff2_l)
        out = ff2_out.view(batch_size, args.n_classes)
        class_probs = F.softmax(out, dim=-1)
        return class_probs
    
    def evaluate(self, test_data, log=sys.stdout):
        self.eval()
        test_batches = batchify_data(test_data[0], test_data[1], shuffle=False)
        ncorr = 0
        ntot = 0
        y_true = []
        y_pred = []
        num_batch = 0
        t1 = datetime.now()
        with torch.no_grad():
            for batch in test_batches:
                node_docs, node_mask, comb_amtx, src_tensor, dest_tensor, y_tensor = batch
                batch_out = self.forward(node_docs, node_mask, comb_amtx, src_tensor, dest_tensor)
                batch_out = batch_out.cpu()
                if torch.isnan(batch_out).any():
                    print('NaN Error', file=log, flush=True)
                y_pred += list(torch.argmax(batch_out, dim=1).data.numpy())
                y_true += list(y_tensor.data.numpy())
                ncorr += sum((torch.argmax(batch_out, dim=1).data == y_tensor.data).int()).item()
                ntot += y_tensor.size(0)
                num_batch += 1
            print('Accuracy: ', round(ncorr * 100./ntot, 2), ncorr, ntot, file=log, flush=True)
            y_true = np.array(y_true, dtype=int)
            y_pred = np.array(y_pred, dtype=int)
            print(confusion_matrix(y_true, y_pred), file=log, flush=True)
            return ncorr * 100./ntot
    
    def train_model(self, train_data, val_data, num_epochs=3, save_path=fpath + 'saved_parameters/link_predictor.pt', log=sys.stdout):
        best_val_perf = 0
        t1 = datetime.now()
        for epoch in range(num_epochs):
            self.train()
            train_batches = batchify_data(train_data[0], train_data[1], shuffle=True)
            num_batch = 0
            for batch in train_batches:
                node_docs, node_mask, comb_amtx, src_tensor, dest_tensor, y_tensor = batch
                batch_out = self.forward(node_docs, node_mask, comb_amtx, src_tensor, dest_tensor)
                if self.args.cuda:
                    y_tensor = y_tensor.cuda()
                batch_loss = self.loss_fn(batch_out, y_tensor)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                num_batch += 1
                if num_batch % 1000 == 0:
                    t2 = datetime.now()
                    print(num_batch, ' done.', t2-t1, file=log, flush=True)
            print('Epoch ', epoch, ' completed - ', t2 - t1, file=log, flush=True)
            val_perf = self.evaluate(val_data, log=log)
            if val_perf >= best_val_perf:
                best_val_perf = val_perf
                print('Saving parameters', file=log, flush=True)
                torch.save(self.state_dict(), save_path)


class Args:
    def __init__(self):
        self.cuda = True,
        self.d_encoder = 768
        self.d_hidden = 384
        self.n_heads = 12
        self.d_k = 64
        self.d_v = 64
        self.n_layers = 2
        self.dropout = 0.1
        self.n_classes = 2
        self.lr = 0.0075
        self.weight_decay = 0
        self.pretrained = None
        self.init_linear = False


args = Args()

#Change this number if restarting after interruption after a certain batch with the next batch to be trained on
already_done = 0


t1 = datetime.now()
query_batches = 100
log = open(fpath + 'training_logs/log_r4056_t4056_phase1.txt', 'a+')
if already_done > 0:
    args.pretrained = fpath + 'saved_parameters/link_predictor_phase1_b' + str(already_done - 1) + '.pt'
else:
    args.init_linear = True
torch.manual_seed(4056)
link_predictor = LinkPredictor(args)
for query_batch in range(already_done, query_batches):
    random.seed(4056)
    qdb_path = fpath + 'data_examples/query_dicts-' + str(query_batch) + '.pkl'
    print('Loading data batch ', query_batch, file=log, flush=True)
    qbeg = query_batch * 24
    qend = (query_batch + 1) * 24
    data = load_data_batch(qdb_path, log=log, trim_auth=True, only_hard=False)
    t2 = datetime.now()
    print(t2 - t1, file=log, flush=True)
    train, dev, test = data
    print('Batch', query_batch, 'Train -', sum(train[1]) / len(train[1]), len(train[1]), file=log, flush=True)
    print('Batch', query_batch, 'Dev -', sum(dev[1]) / len(dev[1]), len(dev[1]), file=log, flush=True)
    print('Batch', query_batch, 'Test -', sum(test[1]) / len(test[1]), len(test[1]), file=log, flush=True)
    with torch.cuda.device(0):
        torch.manual_seed(4056)
        link_predictor.cuda()
        link_predictor.train_model(train, dev, log=log)
        link_predictor.load_state_dict(torch.load(fpath + 'saved_parameters/link_predictor.pt'))
        torch.save(link_predictor.state_dict(), fpath + 'saved_parameters/link_predictor_phase1_b' + str(query_batch) + '.pt')
        print('Final Test Performance', file=log, flush=True)
        link_predictor.evaluate(test, log=log)
    t2 = datetime.now()
    print('Batch', query_batch, 'done -', t2 - t1, file=log, flush=True)
log.close()

