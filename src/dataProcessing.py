# import cheminformatics libraries
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit import DataStructs
# import multifunctional libraries
import os
import numpy as np
import pandas as pd
import string
import pickle
# import image plots
import seaborn as sns
import matplotlib.pyplot as plt
# import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.decomposition import PCA


# set visualization fonts
IPythonConsole.molSize = (400,200)
IPythonConsole.drawOptions.addAtomIndices = True

# use created functions

class DataPreprocessing():
    """

        Contains methods for data acquisition and preprocessing 

    """
    def __init__(self) -> None:
        pass

    def data_load(self, filename: str):
        """
        It takes a file name, a sheet name, and a list of columns as arguments, and returns a dataframe
        
        :param filename: str = name of the file
        :type filename: str
        :param sheetname: The name of the sheet in the excel file, defaults to 0 (optional)
        :param columns: list of columns to be used in the model
        :return: A dataframe
        """

        # Set source as working directory
        try:
            os.chdir('./data/generated')
        except:
            if 'solvation-descriptors' in os.getcwd() or 'activity-coefficient' in os.getcwd():
                os.chdir('../../data/generated')
            else:
                os.chdir('../data/generated')

        if filename.endswith('.xlsx'):
            sheetname=0
            self.data = pd.read_excel(filename, sheet_name=sheetname, encoding='utf-8')

        elif filename.endswith('.csv'):
            self.data = pd.read_csv(filename, encoding='utf-8')

        os.chdir('../../notebooks')

        return self.data
    
    # helper function to compute the Tanimoto similarity between structures
    def computeTanimotoSimilarity(self, dict, method):
        """
        params: dict, a dictionary containing the structure of the different polymers
        return: DataFrame of similarity index
        reference: https://www.programcreek.com/python/example/89502/rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect
        """
        # results
        results = pd.DataFrame({}, columns = dict.keys(), index = dict.keys(), dtype=float)
        
        # fill the result
        if dict is None: 
            return print("dictionary is empty")
        else:
            for (nameA, molA) in dict.items():
                for (nameB, molB) in dict.items():
                    if molA is None or molB is None:
                        results.loc[nameA, nameB] = None
                    else:
                        # get the Morgan Fingerprint
                        fpA = AllChem.GetMorganFingerprintAsBitVect(molA, radius=2, nBits=2048, useChirality=True)
                        fpB = AllChem.GetMorganFingerprintAsBitVect(molB, radius=2, nBits=2048, useChirality=True)
                        # compute the similarity

                        if method == 'Dice':
                            results.loc[nameA, nameB] = round(DataStructs.DiceSimilarity(fpA, fpB), 3)
                        if method == 'Tanimoto':
                            results.loc[nameA, nameB] = round(DataStructs.TanimotoSimilarity(fpA, fpB), 3)
                        
                            
        return results

    # helper function to plot the similarity between structures
    def get_dissimilarity_map(self, df, title, filename):
        """
        params: DataFrame of square size
        return: heat map
        """
        # set fontsize
        plt.rcParams.update({'font.size': 15})
        
        # generate heatmap
        f, ax = plt.subplots(figsize=(15, 10), dpi = 600)
        #
        mask = np.triu(np.ones_like(df, dtype=bool))
        #
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        axs = sns.heatmap(df, mask = mask,  annot = True, fmt = ".2f",center = 0, linewidths = .5)
        #
        cbar = ax.collections[0].colorbar; cbar.ax.tick_params(labelsize=15)
        ax.set_title(title, fontweight="bold", fontsize=30)
        labels = [f'p{i}' for i in string.ascii_uppercase[:len(list(df.columns))]] # based on Figure 4 in the manuscript
        ax.set_xticklabels(labels, rotation = 90, fontsize=15)
        ax.set_yticklabels(labels, rotation = 360, fontsize=15)
        plt.savefig(filename)
        plt.show()

    # helper function to plot the importance between the solvation descriptors and the experimental activity coefficients
    def get_importance_feature_map(self, df, approach):
        """
        params: DataFrame of square size
        return: heat map 
        """
        # import library
        import string
        # set fontsize
        plt.rcParams.update({'font.size': 12})

        # re-order dataframe
        columns = [
                'Water-per-ion', 'concentration_of_salt_M', 'gr_minima_Ion_H2O',
                'gr_peak_position_Ion_H2O', 'gr_peak_height_Ion_H2O', 'Nr_Ion_H2O',
                'gr_minima_CG_H2O', 'gr_peak_position_CG_H2O', 'gr_peak_height_CG_H2O',
                'Nr_CG_H2O_', 'gr_minima_CG_Ion', 'gr_peak_position_CG_Ion',
                'gr_peak_height_CG_Ion', 'Nr_CG_Ion', 'Exp_act_coeff'
                ] 
        
        df = df[columns]
            
        # create figure
        f, axs = plt.subplots(figsize=(14, 12))
        # compute correlation
        importance = df.corr(method = approach)
        mask = np.triu(np.ones_like(importance, dtype=bool))
        # plot map
        map = sns.heatmap(importance, mask = mask,  annot = True, fmt = ".2f",center = 0, linewidths = .5, ax = axs, cmap = 'tab10', cbar_kws={"shrink": 1, "pad": 0.08, "orientation": 'horizontal'}, vmin = -1, vmax = +1)
    
        #
        labels = [f'{i}' for i in string.ascii_uppercase[:len(columns) - 1]] # based on Figure 6 in the manuscript
        labels.append('Exp')
        axs.set_xticklabels(labels, rotation = 90, fontsize = 12)
        axs.set_yticklabels(labels, rotation = 360, fontsize = 12)   
        plt.show()


    def featurizer(self, data, polymers_dict, descr):
        #
        # convert the categorical variable to numerical values via OneHotEncoder
        ions = data[['CounterIon','Co-Ion']].to_numpy()
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(ions)
        ohe_ions = enc.transform(ions).toarray()
        ohe_df = pd.DataFrame(data = ohe_ions, columns=[f'ohe_{i}' for i in range(len(ohe_ions[0]))])
        #
        #****************************************** Generate FingerPrint ************************************************#
        polymers_names = list(polymers_dict.keys())
        #
        # Morgan Fingerprint
        fps_dict = {}
        # generate morgan fingerprints
        for polymer in polymers_names:
            if descr == 'solvation':
                fps_dict[polymer] = AllChem.GetMorganFingerprintAsBitVect(polymers_dict[polymer], useChirality=True, radius=3, nBits=124)
            else:
                fps_dict[polymer] = AllChem.GetMorganFingerprintAsBitVect(polymers_dict[polymer], useChirality=True, radius=3, nBits=128)
        #
        # convert descriptor dictionary into arrays
        vects_dict = {}
        for polymer in polymers_names:
            data_ = np.array(fps_dict[polymer]).reshape(1, -1)
            vects_dict[polymer] = pd.DataFrame(data = data_, columns=[f'mfp_{i}' for i in range(len(data_[0]))])
            #
            # append to dataframe
        
        fingerprints01 = pd.DataFrame()
            #
        for i, polymer in enumerate(data['Name of the polymer']):
            polymer=polymer.replace('\u200b','')
            try:
                fingerprints01 = pd.concat([vects_dict[polymer], fingerprints01.reset_index(drop=True)], axis=0).reset_index(drop=True)
            except KeyError:
                pass
                    #print(fingerprints01)   #
        reverse_fingerprints = fingerprints01.iloc[::-1].reset_index(drop=True)
        #print(fingerprints01)
        


            # drop unneeded columns
        if descr == 'solvation':
            data_m = data.drop(columns = ['#', 'Name of the polymer', 'CounterIon', 'Co-Ion', 'salt',
                                          'Exp_act_coeff'])
            
        
            ## merge data - data01 and machine learning model 1
            data_mm = pd.concat([ohe_df, data_m], axis=1).reset_index(drop=True)
            data_mmm = pd.concat([reverse_fingerprints, data_mm], axis=1).dropna()
               
            x_index01 = ohe_df.shape[1] + reverse_fingerprints.shape[1] + 2 # use 1 if Water-per-ion isn't there
            X01_MF = data_mmm.iloc[:, :x_index01]
            Y01_MF = data_mmm.iloc[:, x_index01:]
        
        if descr == 'activity':
            d01_y=data['Experimental activity co.']
            data_02 = data.drop(columns = ['#', 'Name of the polymer', 'CounterIon', 'Co-Ion', 'salt'])
            #data02_MorganDescr=data_02.replace(r'[^\w\s]|_', '', regex=True)
            #print(data02_MorganDescr)
            min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
            data_scaled = min_max_scaler.fit_transform(data_02)
            data_m = pd.DataFrame(data_scaled)
            data_m.columns = ['A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','EXP']
        
            ## merge data - data01 and machine learning model 1
            data_mm = pd.concat([reverse_fingerprints,ohe_df, data_m], axis=1).reset_index(drop=True)
            #data_mmmm=data_mm.replace(r'[^\w\s]|_', '', regex=True)
            #data_mmmm=data_mm.replace('\u200b', '')
            #print(data_mmmm)
            X01_MF = data_mm.iloc[:, :-1]
            Y01_MF = d01_y
        
                

        ## merge data - data01 and machine learning model 1
        #data_mm = pd.concat([ohe_df, data_m], axis=1).reset_index(drop=True)
        #data_mmm = pd.concat([reverse_fingerprints, data_mm], axis=1).dropna()

        # 
        #x_index01 = ohe_df.shape[1] + reverse_fingerprints.shape[1] + 2 # use 1 if Water-per-ion isn't there
        #X01_MF = data_mmm.iloc[:, :x_index01]
        #y01_MF = data_mmm.iloc[:, x_index01:]
        #X02_MF = data_mmm.iloc[:, :-1]
        #Y02_MF = d01_y

        
        return (X01_MF, Y01_MF, reverse_fingerprints) #return (data_mmm, ohe_ions, reverse_fingerprints)

    # helper function to split and normalize the features (X) and target (y) data 
    def normalizedata(self, x, y, splitRatio, state = 48, transform = False, property = None):
        """
        params: x & y are input and target
        return: if True, normalize x and y else split only.
        """
        
        # split data
        X_train, X_test,y_train, y_test = train_test_split(x, y, random_state = state, test_size = splitRatio, shuffle=True)

        # transform data
        if transform == True:
            # create a scaler
            scaler = MinMaxScaler(feature_range=(-1, 1))
            if property == 'Y':
        
                scaler.fit(y_train)
                # transform data
                y_train = scaler.transform(y_train)
                y_test  = scaler.transform(y_test)
                # reshape to dataframe
                y_train = pd.DataFrame(y_train, columns = y.columns)
                y_test = pd.DataFrame(y_test, columns = y.columns)
                # set index
                y_train = y_train.set_index(X_train.index)
                y_test  = y_test.set_index(X_test.index)

            if property == 'X':
                # fit data
                scaler.fit(X_train)
                # transform data
                X_train = scaler.transform(X_train)
                X_test  = scaler.transform(X_test)
                # reshape to dataframe
                X_train = pd.DataFrame(X_train, columns = x.columns)
                X_test = pd.DataFrame(X_test, columns = x.columns)
                # set index
                X_train = X_train.set_index(y_train.index)
                X_test  = X_test.set_index(y_test.index)


        return (scaler,X_train, X_test, y_train, y_test)


    # helper function to visualize the variance computed using the PCA
    def visualize_PCA(self, x, length, type):
        """ params: dataframe of the inputs, usually train data
            returns: image of variance against number of components"""
       
        # set number_components
        (n_samples, n_features) = x.shape
        num_components = min(n_samples, n_features)
        # apply PCA
        if type == 'MDFP':
            pca_x = PCA(n_components = num_components, random_state=0)
            pca_x.fit(x.iloc[:, :length])
        if type == "ALL":
            pca_x = PCA(n_components = num_components, random_state=0)
            pca_x.fit(x)
            
        #  plot
        fig, axes = plt.subplots(1, 1, figsize=(5, 3), dpi = 100)
        #
        sns.set(style='whitegrid')
            
        axes.plot(np.cumsum(pca_x.explained_variance_ratio_))
        axes.set_xlabel('Components', fontsize = 15); axes.set_ylabel('Variance', fontsize = 15)
        axes.tick_params(axis='x', labelsize = 13); axes.tick_params(axis='y', labelsize = 13)
        
        axes.axvline(linewidth = 1, color = 'r', linestyle = '--', x = 10, ymin = 0, ymax = 1)
        
        axes.grid(False)
        axes.set_xlim([0, 30])
        plt.show()


    # helper function to convert the data based on the optimal PCA number of components
    def convertInputsPCA(self, X_train, X_test, fingerprints, n_comp, type):
        
        if type == 'MDFP':
            # fit
            pca = PCA(n_components=n_comp)
            length_MDFP = fingerprints.shape[1]
            pca.fit(X_train.iloc[:, :length_MDFP])
            # predict
            pca_X01_MF_train_onlyMDFP = pca.transform(X_train.iloc[:, :length_MDFP])
            pca_X01_MF_test_onlyMDFP  = pca.transform(X_test.iloc[:, :length_MDFP])
            # convert to pandas
            pca_X01_MF_train_onlyMDFP_df = pd.DataFrame(data = pca_X01_MF_train_onlyMDFP, columns=[f'pca_{i}' for i in range(len(pca_X01_MF_train_onlyMDFP[0]))])
            pca_X01_MF_test_onlyMDFP_df = pd.DataFrame(data = pca_X01_MF_test_onlyMDFP, columns=[f'pca_{i}' for i in range(len(pca_X01_MF_test_onlyMDFP[0]))])
            # merge frames
            pca_X01_MF_train = pd.concat([pca_X01_MF_train_onlyMDFP_df, X_train.iloc[:, length_MDFP:].reset_index()], axis = 1)
            pca_X01_MF_test = pd.concat([pca_X01_MF_test_onlyMDFP_df, X_test.iloc[:, length_MDFP:].reset_index()], axis = 1)
            # drop the index column
            pca_X01_MF_train.drop(columns= ['index'], inplace=True)
            pca_X01_MF_test.drop(columns= ['index'], inplace=True)

            return pca_X01_MF_train, pca_X01_MF_test
        
        if type == 'All':
            # fit
            pca = PCA(n_components=n_comp)
            pca.fit(X_train)
            # predict
            pca_X01_MF_train_ = pca.transform(X_train)
            pca_X01_MF_test_ = pca.transform(X_test)
            # convert to pandas
            pca_X01_MF_train = pd.DataFrame(data = pca_X01_MF_train_, columns=[f'pca_{i}' for i in range(len(pca_X01_MF_train_[0]))])
            pca_X01_MF_test = pd.DataFrame(data = pca_X01_MF_test_, columns=[f'pca_{i}' for i in range(len(pca_X01_MF_test_[0]))])
            
            return pca_X01_MF_train, pca_X01_MF_test

    def plot(self,model,x_train, y_train,x_test, y_test):
    
        y_predicted_train = model.predict(x_train)
        y_predicted_test = model.predict(x_test)
    
        plt.rcParams['figure.dpi'] = 600
        plt.rcParams['savefig.dpi'] = 600
        fig, ax = plt.subplots(figsize=(6, 5))
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        ax.tick_params(axis="y", which="major",right=False,direction="in",length=5)
        ax.tick_params(axis="x", which="major",direction="in",length=5)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.)  # change width
        plt.legend(loc="upper left",frameon=False)
        plt.title(model,size=8)
        ax.set_xlabel("Activity coefficient $_{(measured)}$",size=12, color = 'black')
        ax.set_ylabel("Activity coefficient $_{(predicted)}$",size=12, color = 'black')
        ax.plot(y_train,y_train,color='black',linewidth=1.5)
        plt.scatter(y_train, y_predicted_train,facecolors='teal',edgecolors='teal',s=25, label="Train",marker = 'x')
        plt.scatter(y_test, y_predicted_test,facecolors='red',edgecolors='gold',s=25, label="Test",marker = 'x')

