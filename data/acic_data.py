from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import os


# load ACIC data with continuous outcome variable and homogeneous treatment effect
def file_names_homo(dgp):
    """
    extract file names for data with a specific data generating number
    dgp: int
        data generating process number
    """
    # note that the relative directory is with respect to the acic_homo.py
    file_names_temp = pd.read_excel('./raw_data/acic/DatasetsCorrespondence.xlsx', header=None)
    file_index = file_names_temp[0].str.contains("CHDScenario"+str(dgp)+"DS", case=True, regex=False)
    file_names = list(file_names_temp[1][file_index][:25])
    return file_names


class acic_data_homo(Dataset):
    """
    Load ACIC data with specific data generating process number
    dgp: int
        data generating process number
    """
    def __init__(self, dgp):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = []
        file_names_list = file_names_homo(dgp)
        root_dir = './raw_data/acic/data_subset'
        for file_name in file_names_list:
            dir = os.path.join(root_dir, file_name + ".csv")
            file = pd.read_csv(dir)
            data.append(file)
        data = pd.concat(data, ignore_index=True)

        self.data_size = len(data.index)
        self.y = torch.FloatTensor(np.array(data['Y'], dtype=np.float32).reshape(self.data_size, 1)).to(device)
        self.treat = torch.FloatTensor(np.array(data['A'], dtype=np.float32)).to(device)
        self.x = torch.FloatTensor(np.array(data.loc[:, ~data.columns.isin(['Y', 'A'])], dtype=np.float32)).to(device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        return y, treat, x


class acic_data_hete(Dataset):
    """
    load ACIC test data that combines different dgp to create heterogeneous treatment effect
    """
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = []
        file_names_list = ['cont2', 'cont3', 'cont6', 'cont7']
        root_dir = './raw_data/acic/test_data'
        for file_name in file_names_list:
            dir = os.path.join(root_dir, file_name + ".csv")
            file = pd.read_csv(dir)
            data.append(file)
        data = pd.concat(data, ignore_index=True)

        self.data_size = len(data.index)
        self.ate = torch.FloatTensor(np.array(data['ATE'], dtype=np.float32)).to(device)
        self.y1 = torch.FloatTensor(np.array(data['EY1'], dtype=np.float32)).to(device)
        self.y0 = torch.FloatTensor(np.array(data['EY0'], dtype=np.float32)).to(device)

        self.y = torch.FloatTensor(np.array(data['Y'], dtype=np.float32).reshape(self.data_size, 1)).to(device)
        self.treat = torch.FloatTensor(np.array(data['A'], dtype=np.float32)).to(device)
        self.x = torch.FloatTensor(np.array(data.loc[:, ~data.columns.isin(['ATE', 'EY1', 'EY0', 'Y', 'A'])],
                                            dtype=np.float32)).to(device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        return y, treat, x
