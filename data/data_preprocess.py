import torch
from torch.utils.data import random_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def data_preprocess(data, partition_seed):
    """
    scale the input variables and the outcome variable to [0,1]
    data: Dataset object
        map-style dataset (only map-style dataset has __len__() property)
    partition_seed: int
        seed to randomly partition the dataset into train set, validation set, and test set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    return train_set, val_set, test_set, x_scaler, y_scaler
