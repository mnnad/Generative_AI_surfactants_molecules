# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:27:52 2020

@author: sqin34
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:54:36 2020

@author: sqin34
"""

# https://docs.dgl.ai/en/0.4.x/tutorials/basics/4_batch.html

import dgl
import torch
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

###############################################################################

class GCNReg(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.

        if torch.cuda.is_available():
            h = g.ndata['h'].float().cuda()
        else:
            h = g.ndata['h'].float()

        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        h1 = F.relu(self.conv2(g, h1))

        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        output = F.relu(self.classify1(hg))
        output = F.relu(self.classify2(output))
        output = self.classify3(output)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output
