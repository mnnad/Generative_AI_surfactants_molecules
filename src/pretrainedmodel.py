import numpy as np
import pandas as pd
import torch
import glob
import os

from .GraphWisconsin.model_GNN import GCNReg
from .GraphWisconsin.generate_graph_dataset import graph_dataset, collate


class GNNmodel:
    def __init__(self, path):
        self.path = path
        self.model = GCNReg(in_dim=74, hidden_dim=256, n_classes=1)


    def get_predictions(self, data_loader):

        y_pred = []
        y_true = []

        for data in data_loader:
            graph = data[0] 
            label = data[1]
                
            label = label.view(-1,1)
            output = self.model(graph)
            y_true.append(label.float())
            y_pred.append(output)


        y_true = [[t.cpu().detach().numpy()[0] for t in sublist][0] for sublist in y_true]

        y_pred = [t.cpu().detach().numpy()[0][0] for t in y_pred if t.grad_fn is not None]
    

        return y_true, y_pred
    
    def avg(self, data):

        first_key = next(iter(data))
        length1 = len(data[first_key][0])
        length2 = len(data[first_key][1])

        sum_list1 = [0] * length1
        sum_list2 = [0] * length2

        # Iterate through the dataframe to compute the sums
        for key in data:
            sum_list1 = [a + b for a, b in zip(sum_list1, data[key][0])]
            sum_list2 = [a + b for a, b in zip(sum_list2, data[key][1])]

        # Calculate the averages
        avg_list1 = [x / len(data) for x in sum_list1]
        avg_list2 = [x / len(data) for x in sum_list2]

        # Return the average tuple
        avg_tuple = (avg_list1, avg_list2)

        return avg_tuple


    def predict(self, smiles, choice_model = 'ep1000bs5lr0.005kf11hu256cvid5'):

        dataset = graph_dataset(smiles = smiles, y = [0]*len(smiles))
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate)


        tar_files = glob.glob(f"{self.path}/{choice_model}*.tar")

        checkpoints = {}
        models = {}
        targets = {}

        for tar_file in tar_files:

            # load checkpoint
            checkpoint = torch.load(tar_file)
            checkpoint_name = os.path.basename(tar_file).replace('.tar', '')
            checkpoints[checkpoint_name] = checkpoint

            # load model with checkpoint
            try:
                self.model.load_state_dict(checkpoint['state_dict'])
            except:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            models[checkpoint_name] = self.model

            self.model.eval()

            targets[checkpoint_name] = self.get_predictions(loader)

        return self.avg(targets)[1]
