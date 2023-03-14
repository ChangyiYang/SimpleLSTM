# Includes the functions for data processing.
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

def read_csv_as_ndarray(file_path: str, skip_rows: list, skip_columns: list) -> np.ndarray:
    '''
    This function reads in data located in FILEPATH and return a ndarray with specific ROWs and COLUMNs skipped.

    In skip_rows/columns, the elements should be the index of the rows/columns want to be skipped.
    '''


    df = pd.read_csv(file_path)

    assert type(skip_columns) == list, "Parameter skip_columns must be a list"
    assert type(skip_rows) == list, "Parameter skip_rows must be a list"

    # drop the specified rows and columns from the DataFrame
    df = df.drop(df.index[skip_rows], axis=0)  # drop rows by index
    df = df.drop(df.columns[skip_columns], axis=1)  # drop columns by index

    return df.values


def standarlize(data : np.ndarray):
    '''
    This function take a np.ndarray() as a input and standarlize it over columns
    '''
    ss = StandardScaler()

    return ss.fit_transform(data)

def generate_histogram(data: np.ndarray, binInt, binMax):
    '''
    Take in a data with first n-1 columns are features and last column are labels. Transform the features with histogram method
    '''

    # Extract data
    Xvals = data[:,0:-1]
    Yvals = data[:,-1]
    nRows, nCols = Xvals.shape

    # define bins
    hEdges = np.arange(0, binMax+binInt, binInt)
    nBins = len(hEdges)-1


    # fill histogram matrix
    nCountsX = np.zeros((nRows, nBins), dtype=np.uint16)
    for n in range(nRows):
        nCountsX[n,:], _ = np.histogram(Xvals[n,:], hEdges)

    # add the labels back
    result = np.concatenate((nCountsX, Yvals[:, np.newaxis]), axis=1)

    return result


# define the dataset classes
class ReactorData(Dataset):
    '''
    Define the pytorch dataset

    The input data must a np.ndarray with last column be the labels and all other columns be the features 

    Note that since the return data is a time sequence data, there maybe some data in the end of dataset is not used.

    '''
    def __init__(self, data, sequence_length = 10, start_percent = 0, end_percent = 1):
        
        
        length = data.shape[0]
        data = data[ int(length * start_percent)  : int(length * end_percent)]
        self.all_data = data
        self.labels = data[:, -1:]
        self.data = data[:, 0:-1]
        self.sequence_length = sequence_length

        self.length = len(self.labels)//self.sequence_length

        # cut the out datas

        self.data = self.data[:self.sequence_length * self.length]

        self.labels = self.labels[:self.sequence_length * self.length]

        self.data = self.data.reshape(( self.length, self.sequence_length, self.data.shape[1]))

        self.labels = self.labels.reshape(( self.length, self.sequence_length,1))
        
    # this initial method take in X and y seperately
    def _init_(self, data, labels, sequence_length = 10, start_percent = 0, end_percent = 1):
        length = data.shape[0]
        data = data[ int(length * start_percent)  : int(length * end_percent)]
        data = labels[ int(length * start_percent)  : int(length * end_percent)]
        
        self.all_data = data
        self.labels = labels
        self.data = data
        self.sequence_length = sequence_length

        self.length = len(self.labels)//self.sequence_length

        # cut the out datas

        self.data = self.data[:self.sequence_length * self.length]

        self.labels = self.labels[:self.sequence_length * self.length]

        self.data = self.data.reshape(( self.length, self.sequence_length, self.data.shape[1]))

        self.labels = self.labels.reshape(( self.length, self.sequence_length,1))
        
    
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        return torch.tensor(self.data[idx], dtype = torch.double), \
    torch.tensor(self.labels[idx], dtype = torch.double)