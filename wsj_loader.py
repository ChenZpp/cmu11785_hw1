import numpy as np
import os
import torch
from torch.utils import data

class WSJ():
    """ Load the WSJ speech dataset
        
        Ensure WSJ_PATH is path to directory containing 
        all data files (.npy) provided on Kaggle.
        
        Example usage:
            loader = WSJ()
            trainX, trainY = loader.train
            assert(trainX.shape[0] == 24590)
            
    """
  
    def __init__(self):
        self.dev_set = None
        self.train_set = None
        self.test_set = None
  
    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = load_raw(os.environ['WSJ_PATH'], 'dev')
        return self.dev_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_raw(os.environ['WSJ_PATH'], 'train')
        return self.train_set
  
    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(os.environ['WSJ_PATH'], 'test.npy'), encoding='bytes'), None)
        return self.test_set
    
def load_raw(path, name):
    return (
        np.load(os.path.join(path, '{}.npy'.format(name)), encoding='bytes'),
        np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes')
    )

class MyDataset(data.Dataset):

    def __init__(self, wsj, k, mydevice):

        self.is_train = (wsj[1]!=None)
        print("________Initializing Data Loader, Train = ",self.is_train,"_________")
        self.MyDevice = mydevice
        self.x = wsj[0]
        self.y = wsj[1] if self.is_train else None
        self.k = k
        self.index = []

        for i, x in enumerate(self.x):
            num_rows = self.x[i].shape[0]
            self.x[i] = np.pad(x,((k,k),(0,0)),'constant',constant_values=(0,0))
            for j in range(num_rows):
                self.index.append((i,j+k))

        print("num of training data:", len(self.index))

    def __getitem__(self, idx):
        i,j = self.index[idx]
        xi = self.x[i].take(range(j - self.k, j + self.k+1), mode='clip', axis=0).flatten()
        xi = torch.from_numpy(xi).float()
        if self.is_train:
            try:
                yi = np.array(self.y[i][j-self.k])
            except:
                print("i:",i,"j:",j,"j-k",j-self.k)
                print(self.y.shape)
                print(self.y[i].shape)
                print(self.x.shape)
                print(self.x[i].shape)
                
            yi = torch.from_numpy(yi).long()
        else:
            yi = torch.from_numpy(np.array(0)).long()
        return xi, yi

    def __len__(self):
        return len(self.index)

