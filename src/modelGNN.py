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


###############################################################################

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
      
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        h = g.ndata['h'].float().cuda()
        # Perform graph convolution and activation function.
        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        h1 = F.relu(self.conv2(g, h1))
        h1 = F.relu(self.conv3(g, h1))

        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        output = F.relu(self.classify1(hg))
        output = F.relu(self.classify2(output))
#        output = F.relu(self.classify2(output))
        output = self.classify3(output)
        
        return output
