# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 01:49:12 2020

@author: sqin34
"""
import torch
from rdkit import Chem
from src.transform_graph import smiles_to_graph

class graph_dataset(object):

    def __init__(self, smiles, y, graph_type = smiles_to_graph):
        super(graph_dataset, self).__init__()
        self.smiles = smiles
        self.y = y
        self.graphs = []
        self.labels = []
        self.graph_type = graph_type
        self._generate()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Return the graph and its label of the i^th sample.
        """
        return self.graphs[idx], self.labels[idx]

    def getsmiles(self, idx):
        return self.smiles[idx]
    
    def _generate(self):
        for i,m in enumerate(self.smiles):
            #m = Chem.MolFromSmiles(j) # since smiles_to_graph() already does this
            g = self.graph_type(m)
            self.graphs.append(g)
            self.labels.append(torch.tensor(self.y[i], dtype=torch.float32))
        