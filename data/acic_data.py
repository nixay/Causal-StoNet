from torchdata.datapipes.iter import FileLister, FileOpener
import pandas as pd
import os
import numpy as np
import functools
import torch


# load ACIC data with continuous variable
def file_names_cont(dgp):
    """
    extract file names for data with a specific data generating number
    dgp: int
        data generating process numbers
    #TO-DO: include data with different dgp to create heterogeneous treatment effect
    """
    file_names_temp = pd.read_excel('./raw_data/acic/DatasetsCorrespondence.xlsx', header=None)
    file_index = file_names_temp[0].str.contains("CHDScenario"+str(dgp)+"DS", case=True, regex=False)
    file_names = list(file_names_temp[1][file_index])
    return file_names


def data_filter(filepath, file_names):
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    return file_name in file_names


def row_processor_cont(row):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y = torch.FloatTensor(np.array(row[0], dtype=np.float32).reshape(1)).to(device)
    treat = torch.FloatTensor(np.array(row[1], dtype=np.float32)).to(device)
    x = torch.FloatTensor(np.array(row[2:], dtype=np.float32)).to(device)
    return y, treat, x


def acic_data_cont(dgp):
    """
    Load ACIC data with specific data generating process number
    dgp: int
        data generating process number
    """

    # select the csv files to be loaded
    file_names_list = file_names_cont(dgp)
    fn = functools.partial(data_filter, file_names=file_names_list)

    # read in csv rows
    data_pipe = FileLister(root='./raw_data/acic/data').filter(filter_fn=fn)
    data_pipe = FileOpener(data_pipe, mode='rt')
    data_pipe = data_pipe.parse_csv(delimiter=',', skip_lines=1)
    data_pipe = data_pipe.shuffle()

    # process csv rows
    data_pipe = data_pipe.map(row_processor_cont)
    return data_pipe


# # load ACIC out-of-sample data
# def data_filter_sample(filepath, data_type):
#     file_name = os.path.splitext(os.path.basename(filepath))[0]
#     return data_type in file_name
#
#
# def row_processor_cont_test(row):
#     ATE = np.array(row[0], dtype=np.float32)
#     EY1 = np.array(row[1], dtype=np.float32)
#     EY0 = np.array(row[2], dtype=np.float32)
#     y = np.array(row[3], dtype=np.float32).reshape(1)
#     treat = np.array(row[4], dtype=np.float32)
#     x = np.array(row[5:], dtype=np.float32)
#     return ATE, EY1, EY0, y, treat, x
#
#
# def row_processor_bin_test(row):
#     ATE = np.array(row[0], dtype=np.float32)
#     EY1 = np.array(row[1], dtype=np.float32)
#     EY0 = np.array(row[2], dtype=np.float32)
#     label = np.array(row[3], dtype=np.int64)
#     treat = np.array(row[4], dtype=np.float32)
#     x = np.array(row[5:], dtype=np.float32)
#     return ATE, EY1, EY0, label, treat, x
#
#
# def acic_data_test(data_type):
#     """
#     load ACIC test data with ATE, EY1, and EY0.
#     Note that for binary outcome variable, only the speeding date has test set.
#     data_type: str
#         the data type of outcome variable.
#         'cont': continuous variable.
#         'bin': binary variable.
#     """
#     fn = functools.partial(data_filter_sample, data_type=data_type)
#
#     # read in csv rows
#     data_pipe = FileLister(root='./test_data').filter(filter_fn=fn)
#     data_pipe = FileOpener(data_pipe, mode='rt')
#     data_pipe = data_pipe.parse_csv(delimiter=',', skip_lines=1)
#     data_pipe = data_pipe.shuffle()
#
#     # process csv rows
#     if data_type == 'cont':
#         data_pipe = data_pipe.map(row_processor_cont_test)
#     if data_type == 'bin':
#         data_pipe = data_pipe.map(row_processor_bin_test)
#     return data_pipe
#
#
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
