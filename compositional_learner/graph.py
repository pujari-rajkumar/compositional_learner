from compositional_learner.node import Node
import torch

__author__ = "Rajkumar Pujari"

class Graph():

    def __init__(self, meta_data={}):
        self.nodes = {}
        self.adj_list = {}
        self.node_idx = 0
        self.meta_data = {}

    def add_node(self, node, edges):
        idx = self.node_idx
        self.node_idx += 1
        self.nodes[idx] = node
        self.adj_list[idx] = edges
        for c_node in edges:
            self.adj_list[c_node].append(idx)

    def get_adjacency_matrix(self):
        n = self.node_idx
        adj_mtx = torch.zeros(n, n)
        for n_idx in self.adj_list:
            for c_idx in self.adj_list[n_idx]:
                adj_mtx[n_idx][c_idx] = 1
        return adj_mtx

        

            

