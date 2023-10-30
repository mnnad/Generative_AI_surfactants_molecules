import random, pickle, csv
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def read_data(csv_path="../data/dataset.csv", seed=2023, test_size=0.2, randSplit=True, write_out_files=False):
    
    # fix random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # load CSV dataset
    smlstr = []
    prop = []
    with open(csv_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            smlstr.append(row[0])
            prop.append(row[1])
    smlstr = np.asarray(smlstr)
    prop = np.asarray(prop, dtype="float")
    dataset_size = len(smlstr)
    all_ind = np.arange(dataset_size)

    # split into training and testing
    if randSplit:
        print("Using Random Splits")
        train_ind, test_ind, \
        smlstr_train, smlstr_test, \
        prop_train, prop_test = train_test_split(all_ind, smlstr, prop,
                                                     test_size=test_size,
                                                     random_state=seed)   
        
    else:
        print("no option for non-random splits yet")

    
    # save train/test data and index corresponding to the original dataset
    if write_out_files:
        pickle.dump(smlstr_train,open("../data/smlstr_train.p","wb"))
        pickle.dump(smlstr_test,open("../data/smlstr_test.p","wb"))
        pickle.dump(prop_train,open("../data/prop_train.p","wb"))
        pickle.dump(prop_test,open("../data/lprop_test.p","wb"))
        pickle.dump(train_ind,open("../data/original_ind_train_full.p","wb"))
        pickle.dump(test_ind,open("../data/original_ind_test.p","wb"))
        rows = zip(train_ind,smlstr_train,prop_train)
        with open("../data/dataset_train.csv",'w',newline='') as f:
            writer = csv.writer(f,delimiter=',')
            for row in rows:
                writer.writerow(row)
        rows = zip(test_ind,smlstr_test,prop_test)
        with open("../data/dataset_test.csv",'w',newline='') as f:
            writer = csv.writer(f,delimiter=',')
            for row in rows:
                writer.writerow(row)

    print("Train/Test split complete. See ../data/ for output files (if desired).")

    return [train_ind, test_ind, smlstr_train, smlstr_test, prop_train, prop_test]