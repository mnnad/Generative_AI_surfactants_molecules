#!/bin/bash

python ./GraphWisconsin/GNN_workflow.py --gpu -1 --train --randSplit --path '../models/GCN_early_stop' --data '../data/test_more_dataNoinionc.csv' --seed 2024 --test_size 0.2 --epochs 1000 --early_stop
