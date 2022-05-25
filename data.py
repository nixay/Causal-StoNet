from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import torch


class ACICdata_Cont(Dataset):
    """
    This class corresponds to the dataset with continuous outcome variable
    There are 32 unique data generating processes, each corresponds to 100 .csv files
    one .csv file will be loaded at a time (hence one batch corresponds to a .csv file in dataloader)

    dgp_no: float
            specify the data generating processes that generate the data.
            range of dgp_no: 1-16
    """

    def __init__(self, dgp_no):
        self.data_dir = os.path.join(os.getcwd(), 'data')
        # extract the datasets that correspond to the specified dgp
        filenames_dir = os.path.join(self.data_dir, 'DatasetsCorrespondence.xlsx')
        file_names = pd.read_excel(filenames_dir, header=None, usecols='B', skiprows=100 * (dgp_no - 1) + 1,
                                   nrows=100).values.tolist()
        # flatten the list of filenames to make it a 1D iterable
        self.file_names = [item for sublist in file_names for item in sublist]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        filename = str(self.file_names[index])
        file_dir = os.path.join(self.data_dir, filename + '.csv')
        data = pd.read_csv(file_dir)
        y = np.array(data.loc[:, 'Y'])
        a = np.array(data.loc[:, 'A'])
        x = np.array(data.loc[:, 'V1':'V200'])
        return y, a, x

    # consider datapipes for loading multiple csv files
    # see website: https://pytorch.org/data/main/tutorial.html
    # will need to further decompose data into minibatches


class SimulationData_Cont(Dataset):
    """
    simulation dataset for regression task

    seed: float
        random seed
    size: float
        size of the simulation dataset
    """
    def __init__(self, seed, size):
        self.size = size
        self.seed = seed
        self.x, self.y = [], []

    def generate_data(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        sigma = 1.0
        for i in range(int(self.size)):
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            x_temp = np.repeat((ee + np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(2.0), 5)
            y_temp = 5 * x_temp[1] / (1 + x_temp[0] * x_temp[0]) + 5 * np.sin(x_temp[2] * x_temp[3]) + 2 * x_temp[
                4] + np.random.normal(0, 1)
            y_temp = np.reshape(y_temp, 1)  # reshape y to make it fit for the dimension of network output
            self.x.append(x_temp.astype('float32'))
            self.y.append(y_temp.astype('float32'))
        return self.y, self.x

    def __len__(self):
        return int(self.size)

    def __getitem__(self, idx):
        self.generate_data()
        y = self.y[idx]
        x = self.x[idx]
        return y, x

class SimulationData_Bin(Dataset):
    """
    simulation dataset for classification task.

    seed: float
        random seed
    size: float
        size of the simulation dataset
    """
    def __init__(self, seed, size):
        self.size = size
        self.seed = seed
        self.x, self.label = [], []

    def generate_data(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        sigma = 1.0
        for i in range(int(self.size)):
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            x_temp = np.repeat((ee + np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(2.0), 5)
            score = 5 * x_temp[1] / (1 + x_temp[0] * x_temp[0]) + 5 * np.sin(x_temp[2] * x_temp[3]) + 2 * x_temp[
                4] + np.random.normal(0, 1)
            prob = np.exp(score) / (1 + np.exp(score))
            if prob >= 0.5:
                label_temp = 1
            else:
                label_temp = 0
            self.x.append(x_temp.astype('float32'))
            self.label.append(label_temp)
        return self.label, self.x

    def __len__(self):
        return int(self.size)

    def __getitem__(self, idx):
        self.generate_data()
        label = self.label[idx]
        x = self.x[idx]
        return label, x
