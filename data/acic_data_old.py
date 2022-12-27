from torchdata.datapipes.iter import FileLister, FileOpener
from torchtext.data import to_map_style_dataset
from torch.utils.data import random_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import numpy as np
import functools
import torch


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


def data_filter(filepath, file_names):
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    return file_name in file_names


def row_processor_homo(row):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y = torch.FloatTensor(np.array(row[0], dtype=np.float32).reshape(1)).to(device)
    treat = torch.FloatTensor(np.array(row[1], dtype=np.float32)).to(device)
    x = torch.FloatTensor(np.array(row[2:], dtype=np.float32)).to(device)
    return y, treat, x


def acic_data_homo(dgp):
    """
    Load ACIC data with specific data generating process number
    dgp: int
        data generating process number
    """
    # select the csv files to be loaded
    file_names_list = file_names_homo(dgp)
    fn = functools.partial(data_filter, file_names=file_names_list)

    # read in csv rows
    # note that the relative directory is with respect to the acic_homo.py
    data_pipe = FileLister(root='./raw_data/acic/data_subset').filter(filter_fn=fn)
    data_pipe = FileOpener(data_pipe, mode='rt')
    data_pipe = data_pipe.parse_csv(delimiter=',', skip_lines=1)
    data_pipe = data_pipe.shuffle()

    # process csv rows
    data_pipe = data_pipe.map(row_processor_homo)
    return data_pipe


def data_preprocess_homo(dgp, partition_seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = to_map_style_dataset(acic_data_homo(dgp))
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


# load ACIC data with continuous outcome variable and heterogeneous treatment effect
def row_processor_hete(row):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    true_ate = torch.FloatTensor(np.array(row[0], dtype=np.float32)).to(device)
    y = torch.FloatTensor(np.array(row[3], dtype=np.float32).reshape(1)).to(device)
    treat = torch.FloatTensor(np.array(row[4], dtype=np.float32)).to(device)
    x = torch.FloatTensor(np.array(row[5:], dtype=np.float32)).to(device)
    return true_ate, y, treat, x


def acic_data_hete():
    """
    load ACIC test data that combines different dgp to create heterogeneous treatment effect
    """
    # select the csv files to be loaded
    file_names_list = ['cont2', 'cont3', 'cont6', 'cont7']
    fn = functools.partial(data_filter, file_names=file_names_list)

    # read in csv rows
    # note that the relative directory is with respect to the acic_hete.py
    data_pipe = FileLister(root='./raw_data/acic/test_data').filter(filter_fn=fn)
    data_pipe = FileOpener(data_pipe, mode='rt')
    data_pipe = data_pipe.parse_csv(delimiter=',', skip_lines=1)
    data_pipe = data_pipe.shuffle()

    # process csv rows
    data_pipe = data_pipe.map(row_processor_hete)
    return data_pipe

# # for binary outcome variable
# def file_names_bin(data_source):
#     """
#     extract file names with binary outcome variable based on their data name
#     data_source: string
#         data name of the original dataset.
#         binary_data_souce.csv is a helper csv file
#     """
#     file_names_temp = pd.read_csv('./binary_data_source.csv', header=None, delimiter=',')
#     file_index = file_names_temp[1].str.contains(data_source, case=True, regex=False)
#     file_names = list(file_names_temp[0][file_index])
#     return file_names
#
#
# def row_processor_bin(row):
#     label = np.array(row[0], dtype=np.int64)
#     treat = np.array(row[1], dtype=np.float32)
#     x = np.array(row[2:], dtype=np.float32)
#     return label, treat, x
#
#
# def acic_data_bin(data_name):
#     """
#     Load ACIC data with binary outcome variable. Note that this dataset by nature has heterogeneous treatment effect
#     data_name: str
#         For data with binary outcome variable.
#         Specify the data source.
#         'speed': the speed dating dataset.
#         'epi': the epilepsy dataset.
#     """
#     # select the csv files to be loaded
#     file_names_list = file_names_bin(data_name)
#     fn = functools.partial(data_filter, file_names=file_names_list)
#
#     # read in csv rows
#     data_pipe = FileLister(root='./data').filter(filter_fn=fn)
#     data_pipe = FileOpener(data_pipe, mode='rt')
#     data_pipe = data_pipe.parse_csv(delimiter=',', skip_lines=1)
#     data_pipe = data_pipe.shuffle()
#
#     # process csv rows
#     data_pipe = data_pipe.map(row_processor_bin)
#     return data_pipe
