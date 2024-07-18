import numpy as np
import pandas as pd
from datetime import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import interp1d as i1
from scipy.spatial import distance
from scipy.linalg import inv
import scipy.stats as s
from scipy.stats import gaussian_kde as kde

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.covariance import EllipticEnvelope as EE
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA as ICA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FactorAnalysis

class DFE_object:
    
    def __init__(self):
        self.data_in = {}
        self.window_scheme = None
        self.history_log = []
        self.active_dataset = None
        self.metadata = {'t': None, 'y': None, 'categorical': []} # this is metadata for the active dataset, not for data_in
        self.window_scheme = {'isSet': False, 'window_length': None, 'window_overlap': None}
        
    def import_from_pandas(self, df, **kwargs):
        
        # Check User Inputs
        assert isinstance(df, pd.DataFrame), "Should be Pandas DataFrame."
        for key in kwargs.keys():
            if key.lower() in ['time', 't', 'timecol', 'time_col', 'time_column', 'time column', 'time col']:
                assert isinstance(kwargs[key], int), "Time Column (timeCol) should be an integer."
            elif key.lower() in ['y', 'output', 'ycol', 'y_col', 'ycolumn', 'y_column', 'y col', 'y column', 'output_column']:
                assert isinstance(kwargs[key], int), "Y Column (y) should be an integer."
                
        new_dict = {}
        new_dict['categorical'] = []
        
        # Import Dataset
        df.reset_index(drop=True) # remove index
        #new_dict['raw_data'] = df
        
        # add metadata
        named = False
        for key in kwargs.keys():
            if key.lower() == "name":
                name = kwargs[key]
                named = True
            elif key.lower() in ['time', 't', 'timecol', 'time_col', 'time_column', 'time column', 'time col']:
                new_dict['t'] = kwargs[key]
            elif key.lower() in ['y', 'output', 'ycol', 'y_col', 'ycolumn', 'y_column', 'y col', 'y column', 'output_column']:
                new_dict['y'] = kwargs[key]
            elif key.lower() in ['categorical', 'categorical column', 'categorical columns', 'cats', 'catcols', 'cat_cols', 'catcol', 'cat_col']:
                new_dict['categorical'].append(kwargs[key])
            # keep open for future metadata as needed
        
        if not named:
            name = 'Entry_' + str(len(self.data_in)) 
        if not "t" in new_dict.keys():
            new_dict['t'] = None
        if not "y" in new_dict.keys():
            new_dict['y'] = None

        # Drop Non-Numeric Columns after Category Columns identified
        for cat_col in new_dict['categorical']:
            #new_dict['raw_data'][cat_col] = new_dict['raw_data'][cat_col].astype('category').cat.codes
            df[cat_col] = df[cat_col].astype('category').cat.codes
        #df = df.select_dtypes(include=[np.number])
        df = df.apply(pd.to_numeric)
        new_dict['raw_data'] = df
                        
        # add other metadata that is not set by the user
        new_dict['window_length'] = None
        new_dict['window_overlap'] = None
        new_dict['parent_dataset'] = None
        new_dict['feature'] = None

        self.data_in[name] = new_dict
        history_entry = f"Added dataset, {name}, at {dt.now()}."
        self.history_log.append(history_entry) 

    def drop_dataset(self, dataset_name):
        # removes a raw dataset given its name
        
        # Check User Inputs
        assert isinstance(dataset_name, str), "Name of dataset should be a string."
        
        try:
            self.data_in.pop(dataset_name)
        except:
            print(f"Could not find dataset named {dataset_name}. Cancelling operation.")        

##################
### Visibility ###
##################

    def see_datasets(self):
        # shows the raw datasets by name
        print(self.data_in.keys())
        return self.data_in.keys()
    
    def see_logs(self):
        for i in range(len(self.history_log)):
            print(self.history_log[i])
        return self.history_log
    
#######################
###  Data Alignment ###
#######################
        
    def temporal_alignment(self, universal_time):
        
        # Check User Input 
        assert isinstance(universal_time, pd.DataFrame), "Universal time should be a pandas dataframe."
        
        for i in range(len(self.data_in)):
            
            ret_data = np.array(universal_time)
            dataset_name = list(self.data_in.keys())[i]
            dataset = self.data_in[dataset_name]['raw_data']
            
            col_names = ['time']
            cat_indexes = self.data_in[dataset_name]['categorical']
            
            if self.data_in[dataset_name]['t'] == None:
                print(self.data_in[dataset_name]['name'] + " contains no time column and cannot be temporally aligned. Skipping")
                continue
            else:
                time_col = self.data_in[dataset_name]['t']

            [I,J] = np.shape(dataset)
            old_x = dataset[time_col]
            new_dataset = np.zeros((len(ret_data),J-1))

            new_col_i = 0
            for j in range(J):
                if not j == time_col:
                    if not j in cat_indexes:

                        old_col = dataset[j]
                        f = i1(old_x, old_col, kind="linear", fill_value="extrapolate")
                        new_col = f(universal_time).reshape(-1,1)
                        new_dataset[:,new_col_i] = new_col.flatten()
                        new_col_i += 1
                        
                    elif j in cat_indexes:

                        old_col = dataset[j]
                        f_act = i1(old_x,old_col,kind="nearest",bounds_error=False,fill_value="extrapolate") #kind="previous" would be more desirable, but can't get it working
                        new_col = f_act(universal_time).reshape(-1,1)
                        new_dataset[:,new_col_i] = new_col.flatten()
                        new_col_i += 1
                        
            ret_data = np.concatenate((ret_data, new_dataset), axis = 1)
            ret_data = pd.DataFrame((ret_data))
            self.data_in[dataset_name]['raw_data'] = ret_data
        
            history_entry = f"Temporally aligned dataset {dataset_name} at {dt.now()}."
            self.history_log.append(history_entry)
            
    def KDE(self, dataset_name, t_new):
    # kernel density estimation

        # check user input
        assert isinstance(dataset_name, str), "Dataset name should be a string."
        assert dataset_name in self.data_in.keys(), f"Unrecognized dataset name: {dataset_name}."
        
        dataset = np.array(np.unique(self.data_in[dataset_name]['raw_data'])) # not sure why unique is needed here
        kde_model = kde(dataset)
        y = kde_model(t_new)
        ret = np.concatenate((t_new.reshape((len(t_new),1)),y.reshape((len(y),1))), axis = 1)
        self.data_in[dataset_name]['raw_data'] = pd.DataFrame(ret)
        # fix metadata
        self.data_in[dataset_name]['t'] = 0
        y_col = self.data_in[dataset_name]['y']
        if not y_col == None:
            self.data_in[dataset_name]['y'] = y_col + 1
        
        history_entry = f"Performed KDE on {dataset_name} at {dt.now()}."
        self.history_log.append(history_entry)        
        return 
        
###################
### Data Fusion ###
###################

#    def weighted_Average(self, classifications, weights):
#        # with N being the number of sources,
#        # given classifications as P x N and
#        # given weights as N x 1
#        # determines the classification for each P returned as P x 1
#
#        if self.data_in == []:
#            print("No data has been entered. Add datasets using either import_data_from_path or import_from_pandas.")
#            return
#
#        enc = ohe()
#        elements = np.unique(classifications)
#        enc.fit(elements.reshape(-1,1))
#
#        [P, N] = np.shape(classifications)
#        C = len(elements)
#
#        selections = np.zeros((P,1))
#        for p in range(P):
#            y = enc.transform(classifications[p,:].reshape(-1,1))
#            score = np.zeros((1,C))
#            for n in range(N):
#                score += weights[n]*y[n,:]
#            selections[p] = elements[np.argmax(score)]
#
#        self.active_dataset = selections
#        return

    def concatenate(self, **kwargs):
        # designed to concatenate features together
        # the active_dataset will inherit one time column from the data_in datasets and the y column
        # should be able to specify a window_scheme for selecting which data to concatenate; defaults to current window_scheme

        # Check user input
        if self.data_in == {}:
            print("No data has been entered. Add datasets using import_from_pandas.")
            return

        win_len_provided = False
        win_overlap_provided = False
        for key in kwargs.keys():
            if key.lower() in ['dataset', 'datasets']: # by user list
                assert isinstance(kwargs[key], list), "datasets should be a list of names."
                valid_names = []
                for dataset_name in kwargs[key]:
                    if not isinstance(dataset_name, str): print("dataset names should be strings.")
                    if not dataset_name in self.data_in.keys(): print(f"Could not find dataset {dataset_name}. Skipping")
                    else: valid_names.append(dataset_name)
                self.__concat(valid_names)
                history_entry = f"Concatenated datasets together, producing the active dataset at {dt.now()}."
                self.history_log.append(history_entry)
                return
            elif key.lower() in ['length', 'len', 'window length', 'winlength', 'winlen', 'window_length', 'win_length', 'win_len']:
                assert isinstance(kwargs[key], int), "Window Length (len) should be an integer."
                window_length = kwargs[key]
                win_len_provided = True                
            elif key.lower() in ['overlap', 'win_overlap', 'win overlap', 'window overlap']:
                assert isinstance(kwargs[key], int), "Window Overlap (overlap) should be an integer."
                window_overlap = kwargs[key]
                win_overlap_provided = True

        if win_len_provided and win_overlap_provided:
            valid_names = []
            for dataset_name in self.data_in.keys():
                if not (self.data_in[dataset_name]['window_length'] == window_length) or not (self.data_in[dataset_name]['window_overlap'] == window_overlap):
                    # this is a case of a window mismatch, skip it
                    continue
                else:
                    valid_names.append(dataset_name)
            self.__concat(valid_names)
            history_entry = f"Concatenated datasets together, producing the active dataset at {dt.now()}."
            self.history_log.append(history_entry)
            return

        if self.window_scheme['isSet']:
            valid_names = []
            for dataset_name in self.data_in.keys():
                temp_window_len  = self.data_in[dataset_name]['window_length']
                temp_window_over = self.data_in[dataset_name]['window_overlap']
                if temp_window_len == self.window_scheme['window_length'] and temp_window_over == self.window_scheme['window_overlap']:
                    valid_names.append(dataset_name)
            self.__concat(valid_names)
            history_entry = f"Concatenated datasets together, producing the active dataset at {dt.now()}."
            self.history_log.append(history_entry)
            return
        else:
            self.__concat(list(self.data_in.keys()))
            history_entry = f"Concatenated datasets together, producing the active dataset at {dt.now()}."
            self.history_log.append(history_entry)
            return

    def __concat(self, dataset_names):

        #time_picked  = False
        y_col_picked = False
        y_col = None
        categoricals = []
        [height, _] = np.shape(self.data_in[dataset_names[0]]['raw_data']) # get height of first dataset
        ret = np.ones((height,0))

        for dataset_name in dataset_names:

            # check that height is as expected, skip if not
            [this_height, this_width] = np.shape(self.data_in[dataset_name]['raw_data'])
            [ret_height, ret_width] = np.shape(ret)
            if not this_height == height:
                print(f"The dataset {dataset_name} has a different height from other datasets. Skipping")
                continue

            # time_column, y_column, categorical entries
                #collect first instance of time, activate flag on others
                #collect first instance of y_column, ignore all others
                #continuously add to categorical list
                #if has time, remove it and decrement y_column and categorical entries

            has_time = False
            if not self.data_in[dataset_name]['t'] == None:
                #if not time_picked:
                    #t_col = r_width + self.data_in[dataset_name]['t']
                    #time_picked = True
                t_col = self.data_in[dataset_name]['t']
                has_time = True

            if not self.data_in[dataset_name]['y'] == None:
                if not y_col_picked:
                    if has_time:
                        y_col = ret_width + self.data_in[dataset_name]['y'] - 1
                    else:
                        y_col = ret_width + self.data_in[dataset_name]['y']
                    y_col_picked = True

            for ele in self.data_in[dataset_name]['categorical']:
                if has_time:
                    categoricals.append(ret_width + ele - 1)
                else:
                    categoricals.append(ret_width + ele)

            dataset = np.array(self.data_in[dataset_name]['raw_data'])
            if has_time:
                dataset = np.delete(dataset,t_col,1)
            ret = np.concatenate((ret, dataset), axis = 1)

        self.active_dataset = pd.DataFrame(ret)
        self.metadata['t'] = None
        self.metadata['y'] = y_col
        self.metadata['categorical'] = categoricals

#######################
### Decision Making ###
#######################

    def __tts(self): # train test split
        # assumes the active dataset has been created
        # assumes the active dataset time and y columns have been labeled (there can be at most one of each)
        # assumes unlabeled columns in the active dataset are xs (input variables)

        try:
            if self.active_dataset == None:
                print("The active dataset has not been created. Cancelling operation.")
                return
            print("here")
        except:
            a = 0 #space filler

        dataset = self.active_dataset
        t_index = self.metadata['t']
        y_index = self.metadata['y']
        #print(t_index)
        #print(y_index)

        [M, N] = np.shape(dataset)    
        if not (t_index == None) and not (y_index == None):
            if t_index == y_index:
                print("The time column cannot also be the y column.")
                return
            elif t_index < y_index: 
                dataset_y = dataset.iloc[:,y_index]
                dataset_x = dataset.drop([t_index, y_index], axis = 1)
            else:
                dataset_y = dataset.iloc[:,y_index]
                dataset_x = dataset.drop([t_index, y_index], axis = 1) 
        elif not t_index == None:
            dataset_x = dataset.drop(t_index, axis = 1)
        elif not y_index == None:
            dataset_y = dataset.iloc[:,y_index]
            dataset_x = dataset.drop(y_index, axis = 1)
            
        X_train, X_test, Y_train, Y_test = train_test_split(dataset_x, dataset_y, test_size=0.2, shuffle=True)
        self.Y_test = Y_test
        return X_train, Y_train, X_test

    def elliptic_Envelope_Classification(self):

        X_train, Y_train, X_test = self.__tts() # train test split

        N = len(X_test)
        classes = np.unique(Y_train)
        N_classes = len(classes)

        distances = np.empty((N, N_classes))
        for i in range(N_classes):
            a_class = X_train[Y_train == classes[i]]  # filter for a specific class
            cov = EE().fit(a_class)                   # get elliptic envelope

            for p in range(N):                        # find test points distance from elliptic envelope
                distances[p,i] = distance.mahalanobis(X_test[p,:], cov.location_, inv(cov.covariance_))

        y_est = np.argmin(distances, axis=1)          # classify
        y_est_classes = classes[y_est]
        self.Y_pred = y_est_classes

        history_entry = f"Performed Elliptic Envelope Classification at {dt.now()}."
        self.history_log.append(history_entry)    

    def naive_bayes(self):

        X_train, Y_train, X_test = self.__tts() # train test split
        model = GaussianNB()
        self.Y_pred = model.fit(X_train, Y_train).predict(X_test)

        history_entry = f"Performed Naive Bayes Classification at {dt.now()}."
        self.history_log.append(history_entry)

    def linear_regression(self):

        X_train, Y_train, X_test = self.__tts() # train test split
        model = lr()
        self.Y_pred = model.fit(X_train,Y_train).predict(X_test)

        history_entry = f"Performed Linear Regression Classification at {dt.now()}."
        self.history_log.append(history_entry)

    def LASSO(self, alpha):

        X_train, Y_train, X_test = self.__tts() # train test split
        model = Lasso(alpha = alpha)
        self.Y_pred = model.fit(X_train,Y_train).predict(X_test)

        history_entry = f"Performed LASSO Classification at {dt.now()}."
        self.history_log.append(history_entry)

    def my_LDA(self, n_components = 5):
        # LDA is a dimension reduction and classification; it should only be performed on the active set

        # Check User Input
        assert isinstance(n_components, int), "Number of components (n_components) must be an integer."    
    
        # LDA
        X_train, Y_train, X_test = self.__tts() # train test split
        model = lda(n_components = n_components)
        self.Y_pred = model.fit(X_train,Y_train).predict(X_test)
        
        history_entry = f"Performed LDA Classification at {dt.now()}."
        self.history_log.append(history_entry)

    def random_forest(self):

        X_train, Y_train, X_test = self.__tts() # train test split
        model = RF(max_depth = 5)
        self.Y_pred = model.fit(X_train,Y_train).predict(X_test)

        history_entry = f"Performed Random Forest Classification at {dt.now()}."
        self.history_log.append(history_entry)

    def classification_report(self, performance_metrics = False): # performance metrics are time and space complexity

        print(metrics.classification_report(self.Y_test, self.Y_pred))
        my_confusion_matrix = metrics.confusion_matrix(self.Y_test, self.Y_pred)
        cm = metrics.ConfusionMatrixDisplay(confusion_matrix = my_confusion_matrix)
        cm.plot()
        plt.show()
        return my_confusion_matrix, cm

    def regression_report(self):

        # r
        r = np.corrcoef(np.array(self.Y_test).flatten(), np.array(self.Y_pred.flatten()))
        r = round(r[0,1],2)
        #print("Correlation Coefficient: " + str(r_squared))

        # root mean square error
        rmse = round(np.sqrt(np.mean((self.Y_test - self.Y_pred)**2)),2)
        #print("RMSE: " + str(rmse))

        # relative root mean square error
        rrmse = round(np.sqrt(np.mean((self.Y_test - self.Y_pred)**2)/(np.sum(self.Y_pred**2))),2)
        #print("Rel RMSE: " + str(rrmse))

        # mean absolute error
        mae = round(np.mean(np.abs(self.Y_test - self.Y_pred)),2)
        #print("MAE: " + str(mae))

        # relative absolute error
        yBar = np.mean(self.Y_test)
        rae = round(np.sum(np.abs(self.Y_test - self.Y_pred))/np.sum(np.abs(self.Y_test - yBar)),2)
        #print("Rel MAE: " + str(rae))

        fig,ax = plt.subplots()
        ax.plot(self.Y_test,self.Y_pred,'.')
        ax.axline((0,0),slope=1)
        plt.xlabel('Y test')
        plt.ylabel('Y pred')

        return {'R':r, 'RMSE':rmse, 'RelRMSE':rrmse, 'MAE':mae, 'RAE':rae}

###########################
### Dimension Reduction ###
###########################

    # dataset_name can be any of the raw dataset names, the active dataset, or 'all'

    def my_PCA(self, dataset_name, n_components = 5):

        # Check User Input
        assert isinstance(n_components, int), "Number of components (n_components) must be an integer."
        assert isinstance(dataset_name, str), "Dataset name (dataset_name) must be a string (alternatively, specify 'active' to perform on active set or 'all' to perform on all raw datasets)."

        pca = PCA(n_components)
        if dataset_name == "active":
            dataset = np.array(self.active_dataset)
            y_col_index = self.metadata['y']
            t_col_index = self.metadata['t']
            if not y_col_index == None and t_col_index == None: # PCA should be performed on X-part only, not entire active dataset
                y_column = dataset[:,y_col_index]
                y_column = y_column.reshape((len(y_column),1))
                dataset = np.delete(dataset,y_col_index,1)
                pca.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(pca.components_),y_column), axis = 1)                
                self.active_dataset = pd.DataFrame(ret)
                self.metadata['y'] = n_components
                self.metadata['categorical'] = []
            elif y_col_index == None and not t_col_index == None:
                t_column = dataset[:,t_col_index]
                t_column = t_column.reshape((len(t_column),1))
                dataset = np.delete(dataset,t_col_index,1)
                pca.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(pca.components_),t_column), axis = 1)                
                self.active_dataset = pd.DataFrame(ret)
                self.metadata['t'] = n_components
                self.metadata['categorical'] = []
            elif not y_col_index == None and not t_col_index == None:
                y_column = dataset[:,y_col_index]
                y_column = y_column.reshape((len(y_column),1))
                t_column = dataset[:,t_col_index]
                t_column = t_column.reshape((len(t_column),1))
                dataset = np.delete(dataset,[y_col_index,t_col_index],1)
                pca.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(pca.components_),t_column,y_column), axis = 1)                
                self.active_dataset = pd.DataFrame(ret)
                self.metadata['y'] = n_components 
                self.metadata['t'] = n_components - 1
                self.metadata['categorical'] = []                
            else:
                pca.fit(dataset)
                ret = np.transpose(pca.components_)
                self.active_dataset = pd.DataFrame(ret)
                self.metadata['categorical'] = []

            history_entry = f"Performed PCA on active dataset at {dt.now()}."
            self.history_log.append(history_entry)
            return

        elif dataset_name == "all":
            
            for entry in self.data_in:
                dataset = np.array(self.data_in[entry]['raw_data'])
                y_col_index = self.data_in[entry]['y']
                t_col_index = self.data_in[entry]['t']
                if not y_col_index == None and t_col_index == None: # PCA should be performed on X-part only, not entire active dataset
                    y_column = dataset[:,y_col_index]
                    y_column = y_column.reshape((len(y_column),1))
                    dataset = np.delete(dataset,y_col_index,1)
                    pca.fit(np.transpose(dataset))
                    ret = np.concatenate((np.transpose(pca.components_),y_column), axis = 1)                
                    self.data_in[entry]['raw_data'] = pd.DataFrame(ret)
                    self.data_in[entry]['y'] = n_components
                    self.data_in[entry]['categorical'] = []
                elif y_col_index == None and not t_col_index == None:
                    t_column = dataset[:,t_col_index]
                    t_column = t_column.reshape((len(t_column),1))
                    dataset = np.delete(dataset,t_col_index,1)
                    pca.fit(np.transpose(dataset))
                    ret = np.concatenate((np.transpose(pca.components_),t_column), axis = 1)                
                    self.data_in[entry]['raw_data'] = pd.DataFrame(ret)
                    self.data_in[entry]['t'] = n_components
                    self.data_in[entry]['categorical'] = []                    
                elif not y_col_index == None and not t_col_index == None:
                    y_column = dataset[:,y_col_index]
                    y_column = y_column.reshape((len(y_column),1))
                    t_column = dataset[:,t_col_index]
                    t_column = t_column.reshape((len(t_column),1))
                    dataset = np.delete(dataset,[y_col_index,t_col_index],1)
                    pca.fit(np.transpose(dataset))
                    ret = np.concatenate((np.transpose(pca.components_),t_column,y_column), axis = 1)                
                    self.data_in[entry]['raw_data'] = pd.DataFrame(ret)
                    self.data_in[entry]['y'] = n_components
                    self.data_in[entry]['t'] = n_components - 1
                    self.data_in[entry]['categorical'] = []                     
                else:
                    pca.fit(dataset)
                    ret = np.transpose(pca.components_)
                    self.data_in[entry]['raw_data'] = pd.DataFrame(ret)
                    self.data_in[entry]['categorical'] = []
                                
                pca.fit(np.transpose(np.array(entry['raw_dataset'])))
                entry['raw_dataset'] = pd.DataFrame(np.transpose(pca.components_))

                history_entry = f"Performed PCA on dataset {entry} at {dt.now()}."
                self.history_log.append(history_entry)
            return
        
        else:
                        
            try:
                dataset = np.array(self.data_in[dataset_name]['raw_data'])
            except:
                print(f"Could not find dataset named {dataset_name}. Cancelling operation.")
                return
            
            y_col_index = self.data_in[dataset_name]['y']
            t_col_index = self.data_in[dataset_name]['t']
            if not y_col_index == None and t_col_index == None: # PCA should be performed on X-part only, not entire active dataset
                y_column = dataset[:,y_col_index]
                y_column = y_column.reshape((len(y_column),1))
                dataset = np.delete(dataset,y_col_index,1)
                pca.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(pca.components_),y_column), axis = 1)                
                self.data_in[dataset_name]['raw_data'] = pd.DataFrame(ret)
                self.data_in[dataset_name]['y'] = n_components
                self.data_in[dataset_name]['categorical'] = []
            elif y_col_index == None and not t_col_index == None:
                t_column = dataset[:,t_col_index]
                t_column = t_column.reshape((len(t_column),1))
                dataset = np.delete(dataset,t_col_index,1)
                pca.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(pca.components_),t_column), axis = 1)                
                self.data_in[dataset_name]['raw_data'] = pd.DataFrame(ret)
                self.data_in[dataset_name]['t'] = n_components
                self.data_in[dataset_name]['categorical'] = []                    
            elif not y_col_index == None and not t_col_index == None:
                y_column = dataset[:,y_col_index]
                y_column = y_column.reshape((len(y_column),1))
                t_column = dataset[:,t_col_index]
                t_column = t_column.reshape((len(t_column),1))
                dataset = np.delete(dataset,[y_col_index,t_col_index],1)
                pca.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(pca.components_),t_column,y_column), axis = 1)                
                self.data_in[dataset_name]['raw_data'] = pd.DataFrame(ret)
                self.data_in[dataset_name]['y'] = n_components
                self.data_in[dataset_name]['t'] = n_components - 1
                self.data_in[dataset_name]['categorical'] = []      
            else:
                pca.fit(np.transpose(dataset))
                ret = np.transpose(pca.components_)
                self.data_in[dataset_name]['raw_data'] = pd.DataFrame(ret)
                self.data_in[dataset_name]['categorical'] = []
            
            history_entry = f"Performed PCA on dataset {dataset_name} at {dt.now()}."
            self.history_log.append(history_entry)                
            return

    def my_ICA(self, dataset_name, n_components = 5):

        # Check User Input
        assert isinstance(n_components, int), "Number of components (n_components) must be an integer."
        assert isinstance(dataset_name, str), "Dataset name (dataset_name) must be a string (alternatively, specify 'active' to perform on active set or 'all' to perform on all raw datasets)."

        ica = ICA(n_components)
        if dataset_name == "active":
            dataset = np.array(self.active_dataset)
            y_col_index = self.metadata['y']
            t_col_index = self.metadata['t']
            if not y_col_index == None and t_col_index == None: # ICA should be performed on X-part only, not entire active dataset
                y_column = dataset[:,y_col_index]
                y_column = y_column.reshape((len(y_column),1))
                dataset = np.delete(dataset,y_col_index,1)
                ica.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(ica.components_),y_column), axis = 1)                
                self.active_dataset = pd.DataFrame(ret)
                self.metadata['y'] = n_components
                self.metadata['categorical'] = []
            elif y_col_index == None and not t_col_index == None:
                t_column = dataset[:,t_col_index]
                t_column = t_column.reshape((len(t_column),1))
                dataset = np.delete(dataset,t_col_index,1)
                ica.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(ica.components_),t_column), axis = 1)                
                self.active_dataset = pd.DataFrame(ret)
                self.metadata['t'] = n_components
                self.metadata['categorical'] = []
            elif not y_col_index == None and not t_col_index == None:
                y_column = dataset[:,y_col_index]
                y_column = y_column.reshape((len(y_column),1))
                t_column = dataset[:,t_col_index]
                t_column = t_column.reshape((len(t_column),1))
                dataset = np.delete(dataset,[y_col_index,t_col_index],1)
                ica.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(ica.components_),t_column,y_column), axis = 1)                
                self.active_dataset = pd.DataFrame(ret)
                self.metadata['y'] = n_components 
                self.metadata['t'] = n_components - 1
                self.metadata['categorical'] = []                
            else:
                ica.fit(dataset)
                ret = np.transpose(ica.components_)
                self.active_dataset = pd.DataFrame(ret)
                self.metadata['categorical'] = []

            history_entry = f"Performed ICA on active dataset at {dt.now()}."
            self.history_log.append(history_entry)
            return

        elif dataset_name == "all":
            
            for entry in self.data_in:
                dataset = np.array(self.data_in[entry]['raw_data'])
                y_col_index = self.data_in[entry]['y']
                t_col_index = self.data_in[entry]['t']
                if not y_col_index == None and t_col_index == None: # ICA should be performed on X-part only, not entire active dataset
                    y_column = dataset[:,y_col_index]
                    y_column = y_column.reshape((len(y_column),1))
                    dataset = np.delete(dataset,y_col_index,1)
                    ica.fit(np.transpose(dataset))
                    ret = np.concatenate((np.transpose(ica.components_),y_column), axis = 1)                
                    self.data_in[entry]['raw_data'] = pd.DataFrame(ret)
                    self.data_in[entry]['y'] = n_components
                    self.data_in[entry]['categorical'] = []
                elif y_col_index == None and not t_col_index == None:
                    t_column = dataset[:,t_col_index]
                    t_column = t_column.reshape((len(t_column),1))
                    dataset = np.delete(dataset,t_col_index,1)
                    ica.fit(np.transpose(dataset))
                    ret = np.concatenate((np.transpose(ica.components_),t_column), axis = 1)                
                    self.data_in[entry]['raw_data'] = pd.DataFrame(ret)
                    self.data_in[entry]['t'] = n_components
                    self.data_in[entry]['categorical'] = []                    
                elif not y_col_index == None and not t_col_index == None:
                    y_column = dataset[:,y_col_index]
                    y_column = y_column.reshape((len(y_column),1))
                    t_column = dataset[:,t_col_index]
                    t_column = t_column.reshape((len(t_column),1))
                    dataset = np.delete(dataset,[y_col_index,t_col_index],1)
                    ica.fit(np.transpose(dataset))
                    ret = np.concatenate((np.transpose(ica.components_),t_column,y_column), axis = 1)                
                    self.data_in[entry]['raw_data'] = pd.DataFrame(ret)
                    self.data_in[entry]['y'] = n_components
                    self.data_in[entry]['t'] = n_components - 1
                    self.data_in[entry]['categorical'] = []                     
                else:
                    ica.fit(dataset)
                    ret = np.transpose(ica.components_)
                    self.data_in[entry]['raw_data'] = pd.DataFrame(ret)
                    self.data_in[entry]['categorical'] = []
                                
                ica.fit(np.transpose(np.array(entry['raw_dataset'])))
                entry['raw_dataset'] = pd.DataFrame(np.transpose(ica.components_))

                history_entry = f"Performed ICA on dataset {entry} at {dt.now()}."
                self.history_log.append(history_entry)
            return
        
        else:
                        
            try:
                dataset = np.array(self.data_in[dataset_name]['raw_data'])
            except:
                print(f"Could not find dataset named {dataset_name}. Cancelling operation.")
                return
            
            y_col_index = self.data_in[dataset_name]['y']
            t_col_index = self.data_in[dataset_name]['t']
            if not y_col_index == None and t_col_index == None: # ICA should be performed on X-part only, not entire active dataset
                y_column = dataset[:,y_col_index]
                y_column = y_column.reshape((len(y_column),1))
                dataset = np.delete(dataset,y_col_index,1)
                ica.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(ica.components_),y_column), axis = 1)                
                self.data_in[dataset_name]['raw_data'] = pd.DataFrame(ret)
                self.data_in[dataset_name]['y'] = n_components
                self.data_in[dataset_name]['categorical'] = []
            elif y_col_index == None and not t_col_index == None:
                t_column = dataset[:,t_col_index]
                t_column = t_column.reshape((len(t_column),1))
                dataset = np.delete(dataset,t_col_index,1)
                ica.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(ica.components_),t_column), axis = 1)                
                self.data_in[dataset_name]['raw_data'] = pd.DataFrame(ret)
                self.data_in[dataset_name]['t'] = n_components
                self.data_in[dataset_name]['categorical'] = []                    
            elif not y_col_index == None and not t_col_index == None:
                y_column = dataset[:,y_col_index]
                y_column = y_column.reshape((len(y_column),1))
                t_column = dataset[:,t_col_index]
                t_column = t_column.reshape((len(t_column),1))
                dataset = np.delete(dataset,[y_col_index,t_col_index],1)
                ica.fit(np.transpose(dataset))
                ret = np.concatenate((np.transpose(ica.components_),t_column,y_column), axis = 1)                
                self.data_in[dataset_name]['raw_data'] = pd.DataFrame(ret)
                self.data_in[dataset_name]['y'] = n_components
                self.data_in[dataset_name]['t'] = n_components - 1
                self.data_in[dataset_name]['categorical'] = []      
            else:
                ica.fit(np.transpose(dataset))
                ret = np.transpose(ica.components_)
                self.data_in[dataset_name]['raw_data'] = pd.DataFrame(ret)
                self.data_in[dataset_name]['categorical'] = []
            
            history_entry = f"Performed ICA on dataset {dataset_name} at {dt.now()}."
            self.history_log.append(history_entry)                
            return

#    def my_SVD(self, dataset_name):
#
#        # Check User Input
#        assert isinstance(dataset_name, str), "Dataset name (dataset_name) must be a string (alternatively, specify 'active' to perform on active set or 'all' to perform on all raw datasets)."
#
#        svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
#        if dataset_name == "active":
#            svd.fit(self.active_dataset)
#
#            history_entry = f"Performed SVD on active dataset at {dt.now()}."
#            self.history_log.append(history_entry)
#            return
#
#        elif dataset_name == "all":
#            for entry in self.data_in:
#                svd.fit(entry['raw_dataset'])
#
#                history_entry = f"Performed SVD on dataset {entry['name']} at {dt.now()}."
#                self.history_log.append(history_entry)
#            return
#        else:
#            
#            try:
#                dataset = self.data_in[dataset_name]
#            except:
#                print(f"Could not find dataset named {dataset_name}. Cancelling operation.")
#                return
#            
#            svd.fit(dataset)
#            
#            history_entry = f"Performed SVD on dataset {dataset_name} at {dt.now()}."
#            self.history_log.append(history_entry)                
#            return            
#            
#            ## DON"T KNOW WHAT TO OUTPUT FOR THIS TYPE
#            #print(svd.explained_variance_ratio_)
#            #print(svd.explained_variance_ratio_.sum())
#            #print(svd.singular_values_)
#            #print(np.shape(svd.components_))
#            #print(svd.components_)

#    def my_FA(self, dataset_name):
#
#        # Check User Input
#        assert isinstance(dataset_name, str), "Dataset name (dataset_name) must be a string (alternatively, specify 'active' to perform on active set or 'all' to perform on all raw datasets)."
#
#        fa = FactorAnalysis(n_components=5, random_state=0)
#        if dataset_name == "active":
#            fa.fit(self.active_dataset)
#            self.active_dataset = fa.components_
#
#            history_entry = f"Performed FA on active dataset at {dt.now()}."
#            self.history_log.append(history_entry)
#            return
#
#        elif dataset_name == "all":
#            for entry in self.data_in:
#                fa.fit(entry['raw_dataset'])
#                entry['raw_dataset'] = fa.components_
#
#                history_entry = f"Performed FA on dataset {entry['name']} at {dt.now()}."
#                self.history_log.append(history_entry)
#            return
#        else:
#
#            try:
#                dataset = self.data_in[dataset_name]
#            except:
#                print(f"Could not find dataset named {dataset_name}. Cancelling operation.")
#                return
#            
#            fa.fit(dataset)
#            self.data_in[dataset_name]['raw_data'] = fa.components_
#            
#            history_entry = f"Performed FA on dataset {dataset_name} at {dt.now()}."
#            self.history_log.append(history_entry)                
#            return
            
##########################
### Feature Extraction ###
##########################

    def set_window_scheme(self, **kwargs):

        # Check User Inputs
        wasGivenLen = False
        wasGivenOverlap = False
        for key in kwargs.keys():
            if key.lower() in ['length', 'len', 'window length', 'winlength', 'winlen', 'window_length', 'win_length', 'win_len']:
                assert isinstance(kwargs[key], int), "Window Length (len) should be an integer."
                assert kwargs[key] > 0, "Window Length (len) should be positive."
                wasGivenLen = True
                givenLen = kwargs[key]
            elif key.lower() in ['overlap', 'win_overlap', 'win overlap', 'window overlap']:
                assert isinstance(kwargs[key], int), "Window Overlap (overlap) should be an integer."
                assert kwargs[key] > 0, "Window Overlap (overlap) should be positive." 
                wasGivenOverlap = True
                givenOverlap = kwargs[key]
            else:
                print(f"Could not determine the meaning of input variable '{key}'. \nDid you mean 'len' for window length or 'overlap' for window overlap?")
                return
        if givenLen & givenOverlap:
            assert givenLen >= givenOverlap, "Window Overlap should be less than or equal to window length."
        elif givenLen and not (self.window_scheme['window_overlap'] == None):
            assert givenLen >= self.window_scheme['window_overlap'], "Window Overlap should be less than or equal to window length."
        elif givenOverlap and not (self.window_scheme['window_length'] == None):
            assert self.window_scheme['window_length'] >= givenOverlap, "Window Overlap should be less than or equal to window length."
            
        # add window scheme
        for key in kwargs.keys():
            if key.lower() in ['length', 'len', 'window length', 'winlength', 'winlen', 'window_length', 'win_length', 'win_len']:
                self.window_scheme['window_length'] = kwargs[key]
            elif key.lower() in ['overlap', 'win_overlap', 'win overlap', 'window overlap', 'window_overlap']:
                self.window_scheme['window_overlap'] = kwargs[key]
            # keep open for future metadata as needed
            
        if not (self.window_scheme['window_length'] == None) & (not self.window_scheme['window_overlap'] == None):
            self.window_scheme['isSet'] = True
        else:
            self.window_scheme['isSet'] = False


    def __temporal_feature(self, dataset_name, calculation, calculation_name, *args, **kwargs):

        # Check for window_scheme
        if self.window_scheme['isSet'] == False:
            print("The window scheme must be set.")
            return
        
        # Check dataset exists
        try:
            dataset = np.array(self.data_in[dataset_name]['raw_data'])
        except:
            print(f"Could not find dataset named {dataset_name}. Cancelling operation.")
            return
        
        # Check time column identified
        if self.data_in[dataset_name]['t'] == None:
            print(f"Could not find a time column in {dataset_name}. Cancelling operation.")
            return
        
        time_col = self.data_in[dataset_name]['t']
        startT   = dataset[0,time_col]
        endT     = startT + self.window_scheme['window_length']
        EndT     = dataset[len(dataset)-1,time_col]

        [N, M] = np.shape(dataset)
        window_N = int((EndT - startT)/self.window_scheme['window_overlap'])
        ret = np.zeros((window_N,M))
        for win in range(window_N):

            #data_win1 = data[:,time_col] < endT
            #data_win2 = data[:,time_col] > startT
            #data_win = data[data_win1 & data_win2]
            data_win = dataset[(dataset[:,time_col]<endT) & (dataset[:,time_col]>startT),:] # filter for data from only the window

            for col in range(M):
                if col == time_col:
                    continue     
                data_temp = data_win[:,col]
                if np.ndim(data_temp) == 1:
                    data_temp = data_temp.reshape((len(data_temp),1))
                ret[win,col] = calculation(data_temp,*args,**kwargs)
            #ret[win,:] = calculation(np.delete(data_win,time_col,1),*args,**kwargs) # apply calculation of window

            startT = startT + self.window_scheme['window_overlap']
            endT = endT + self.window_scheme['window_overlap']
     
        new_dict = {}                
        ret = np.delete(ret, time_col, 1) # drop time column
        new_dict['raw_data'] = pd.DataFrame(ret)
        new_dict['t'] = None
        if self.data_in[dataset_name]['y'] == None:
            new_dict['y'] = None
        elif self.data_in[dataset_name]['y'] < time_col:
            new_dict['y'] = self.data_in[dataset_name]['y']
        elif self.data_in[dataset_name]['y'] > time_col:
            new_dict['y'] = self.data_in[dataset_name]['y'] - 1
        new_dict['categorical'] = []
        for ele in self.data_in[dataset_name]['categorical']:
            if ele < time_col:
                new_dict['categorical'].append(ele)
            elif ele > time_col:
                new_dict['categorical'].append(ele-1)
        new_dict['window_length'] = self.window_scheme['window_length']
        new_dict['window_overlap'] = self.window_scheme['window_overlap']
        new_dict['parent_dataset'] = dataset_name
        new_dict['feature'] = calculation_name
        
        name = dataset_name + "_" + calculation_name
        self.data_in[name] = new_dict
        self.__drop_na_features(dataset_name + "_" + calculation_name)
        
        history_entry = f"Calculated " + calculation_name + f" on dataset {dataset_name} at {dt.now()}."
        print(history_entry)
        self.history_log.append(history_entry)

    def fe_average(self, dataset_name):
        assert isinstance(dataset_name, str), "Name of dataset should be a string." # check user input
        self.__temporal_feature(dataset_name, np.average, "average")
        return
    
    def fe_variance(self, dataset_name):
        assert isinstance(dataset_name, str), "Name of dataset should be a string." # check user input
        self.__temporal_feature(dataset_name, np.var, "variance")
        return
        
    def fe_skewness(self, dataset_name):
        assert isinstance(dataset_name, str), "Name of dataset should be a string." # check user input        
        self.__temporal_feature(dataset_name, s.skew, "skewness")
        return
        
    def fe_kurtosis(self, dataset_name):
        assert isinstance(dataset_name, str), "Name of dataset should be a string." # check user input           
        self.__temporal_feature(dataset_name, s.kurtosis, "kurtosis")
        return
        
    def fe_peak_count(self, dataset_name):
        assert isinstance(dataset_name, str), "Name of dataset should be a string." # check user input     
        def pc(x):
            [I,J] = np.shape(x)
            counts = np.zeros((1,J))
            for j in range(J):
                count = 0
                for i in range(1,len(x)-1):
                    if x[i,j] > x[i-1,j] and x[i,j] > x[i+1,j]:
                        count += 1
                counts[0,j] = count
            return count
        self.__temporal_feature(dataset_name, pc, "peak_count")
        return
        
    def fe_RMS(self, dataset_name):
        assert isinstance(dataset_name, str), "Name of dataset should be a string." # check user input    
        def rms(x):
            [I, J] = np.shape(x)
            rmss = np.zeros((1,J))
            for j in range(J):
                rmss[0,j] = np.sqrt(np.mean(x[:,j]**2))
            return rmss
        self.__temporal_feature(dataset_name, rms, "RMS")
        return    
    
    def classification_windowing(self, dataset_name):
        # data may have a time column, or if not, assumes point-wise windowing
        assert isinstance(dataset_name, str), "Name of dataset should be a string." # check user input 
        def max_classify(x):
            [I,J] = np.shape(x)
            maxes = np.zeros((1,J))
            for j in range(J):
                arr = x[:,j].astype('int64')
                count_arr = np.bincount(arr)
                try:
                    maxes[0,j] = np.argmax(count_arr)
                except:
                    maxes[0,j] = -1
            return maxes
        self.__temporal_feature(dataset_name, max_classify, "classification")
        return
        
    #def fe_fourier_max(data, window_scheme, time_col = None):

    #def fe_fourier_max_freq(data, window_scheme, time_col = None):
    
    def normalize(self, dataset_name, ignore = [], type='Z'):
        assert isinstance(dataset_name, str), "Name of dataset should be a string." # check user input         
        # Check dataset exists
        try:
            dataset = self.data_in[dataset_name]
        except:
            print(f"Could not find dataset named {dataset_name}. Cancelling operation.")
            return
        
        X = self.data_in[dataset_name]['raw_data']
        col_names = X.columns
        X = np.array(X)
        [N, M] = np.shape(X)

        if type == 'Z':
            for col in range(M):
                if not col in ignore:
                    mean = np.mean(X[:,col])
                    stdev = np.std(X[:,col])
                    for row in range(N):
                        X[row,col] = (X[row,col] - mean)/stdev # try np.apply_along_axis(function, 1, array)

        elif type == 'MinMax':
            for col in range(M):
                if not col in ignore:
                    mini = min(X[:,col])
                    rang = max(X[:,col]) - mini
                    for row in range(N):
                        X[row, col] = (X[row,col] - mini)/rang

        self.data_in[dataset_name]['raw_data'] = pd.DataFrame(X, columns = col_names)

        history_entry = f"Normalized dataset {dataset_name} at {dt.now()}."
        self.history_log.append(history_entry)   
        return

    def __drop_na_features(self, dataset_name):
        X = np.array(self.data_in[dataset_name]['raw_data'])
        [I,J] = np.shape(X)
        for j in range(J-1,-1,-1):
            if np.isnan(X[:,j]).any():
                X = np.delete(X,j,axis=1)
        self.data_in[dataset_name]['raw_data'] = pd.DataFrame(X)
        return
