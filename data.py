from torchdata.datapipes.iter import FileLister, FileOpener
import pandas as pd
import os
import numpy as np
import functools


# Load ACIC data for training and validation
def file_names_cont_subset(dgp):
    file_names_temp = pd.read_excel('./DatasetsCorrespondence.xlsx', header=None)
    file_index = file_names_temp[0].str.contains("CHDScenario"+str(dgp)+"DS", case=True, regex=False)
    file_names = list(file_names_temp[1][file_index])
    return file_names


def file_names_cont():
    file_names = []
    for i in range(1, 1601):
        file_names.append('high'+str(i))
    return file_names


def file_names_bin(data_source):
    file_names_temp = pd.read_csv('./binary_data_source.csv', header=None, delimiter=',')
    file_index = file_names_temp[1].str.contains(data_source, case=True, regex=False)
    file_names = list(file_names_temp[0][file_index])
    return file_names


def data_filter(filepath, file_names):
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    return file_name in file_names


def row_processor_cont(row):
    y = np.array(row[0], dtype=np.float32).reshape(1)
    treat = np.array(row[1], dtype=np.float32)
    x = np.array(row[2:], dtype=np.float32)
    return y, treat, x


def row_processor_bin(row):
    label = np.array(row[0], dtype=np.int64)
    treat = np.array(row[1], dtype=np.float32)
    x = np.array(row[2:], dtype=np.float32)
    return label, treat, x


def acic_data(data_type, data_name):
    """
    Load ACIC data.

    data_type: str
        the data type of outcome variable.
        'cont': continuous variable.
        'bin': binary variable.
    data_name: str
        For data with binary outcome variable.
        Specify the data source.
        'speed': the speed dating dataset.
        'epi': the epilepsy dataset.
    """
    # select the csv files to be loaded
    if data_type == 'cont':
        file_names_list = file_names_cont()
    if data_type == 'bin':
        file_names_list = file_names_bin(data_name)
    fn = functools.partial(data_filter, file_names=file_names_list)

    # read in csv rows
    data_pipe = FileLister(root='./data').filter(filter_fn=fn)
    data_pipe = FileOpener(data_pipe, mode='rt')
    data_pipe = data_pipe.parse_csv(delimiter=',', skip_lines=1)
    data_pipe = data_pipe.shuffle()

    # process csv rows
    if data_type == 'cont':
        data_pipe = data_pipe.map(row_processor_cont)
    if data_type == 'bin':
        data_pipe = data_pipe.map(row_processor_bin)
    return data_pipe


# load ACIC data for test
def data_filter_test(filepath, data_type):
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    return data_type in file_name


def row_processor_cont_test(row):
    ATE = np.array(row[0], dtype=np.float32)
    EY1 = np.array(row[1], dtype=np.float32)
    EY0 = np.array(row[2], dtype=np.float32)
    y = np.array(row[3], dtype=np.float32).reshape(1)
    treat = np.array(row[4], dtype=np.float32)
    x = np.array(row[5:], dtype=np.float32)
    return ATE, EY1, EY0, y, treat, x


def row_processor_bin_test(row):
    ATE = np.array(row[0], dtype=np.float32)
    EY1 = np.array(row[1], dtype=np.float32)
    EY0 = np.array(row[2], dtype=np.float32)
    label = np.array(row[3], dtype=np.int64)
    treat = np.array(row[4], dtype=np.float32)
    x = np.array(row[5:], dtype=np.float32)
    return ATE, EY1, EY0, label, treat, x


def acic_data_test(data_type):
    """
    load ACIC test data with ATE, EY1, and EY0.
    Note that for binary outcome variable, only the speeding date has test set.

    data_type: str
        the data type of outcome variable.
        'cont': continuous variable.
        'bin': binary variable.
    """
    fn = functools.partial(data_filter_test, data_type=data_type)

    # read in csv rows
    data_pipe = FileLister(root='./test_data').filter(filter_fn=fn)
    data_pipe = FileOpener(data_pipe, mode='rt')
    data_pipe = data_pipe.parse_csv(delimiter=',', skip_lines=1)
    data_pipe = data_pipe.shuffle()

    # process csv rows
    if data_type == 'cont':
        data_pipe = data_pipe.map(row_processor_cont_test)
    if data_type == 'bin':
        data_pipe = data_pipe.map(row_processor_bin_test)
    return data_pipe
