from torchdata.datapipes.iter import FileLister, FileOpener
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
    load ACIC test data.
    data_type: str
        the data type of outcome variable.
        'cont': continuous variable.
        'bin': binary variable.
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
