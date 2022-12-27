from torch.utils.data import Dataset, random_split
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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


def data_preprocess(dgp, partition_seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = acic_data_homo(dgp)
    data_size = data.__len__()
    train_size = int(data_size * 0.6)
    val_size = int(data_size * 0.2)
    test_size = int(data_size * 0.2)
    train_set, val_set, test_set = random_split(data, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(partition_seed))

    train_indices = train_set.indices
    val_indices = val_set.indices
    test_indices = test_set.indices

    x_scaler = MinMaxScaler()
    x_scaler.fit(data.x[train_indices])

    data.x[train_indices] = torch.FloatTensor(np.array(x_scaler.transform(data.x[train_indices]))).to(device)
    data.x[val_indices] = torch.FloatTensor(np.array(x_scaler.transform(data.x[val_indices]))).to(device)
    data.x[test_indices] = torch.FloatTensor(np.array(x_scaler.transform(data.x[test_indices]))).to(device)

    y_scaler = MinMaxScaler()
    y_scaler.fit(data.y[train_indices])

    data.y[train_indices] = torch.FloatTensor(np.array(y_scaler.transform(data.y[train_indices]))).to(device)
    data.y[val_indices] = torch.FloatTensor(np.array(y_scaler.transform(data.y[val_indices]))).to(device)
    data.y[test_indices] = torch.FloatTensor(np.array(y_scaler.transform(data.y[test_indices]))).to(device)

    return train_set, val_set, test_set

