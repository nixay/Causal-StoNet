from torch.utils.data import Dataset
from torchdata.datapipes.iter import FileLister, FileOpener
import pandas as pd
import os
import numpy as np
import torch
import functools
import random


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
    treat = np.array(row[1], dtype=np.int64)
    x = np.array(row[2:], dtype=np.float32)
    return y, treat, x


def row_processor_bin(row):
    label = np.array(row[0], dtype=np.int64)
    treat = np.array(row[1], dtype=np.int64)
    x = np.array(row[2:], dtype=np.float32)
    return label, treat, x


def acic_data(data_type, subset, data_name):
    """
    Load ACIC data.

    data_type: str
        the data type of outcome variable.
        'cont': continuous variable.
        'bin': binary variable.
    subset: bool
        For data with continuous outcome variable.
        When set to be true, a subset of data will used to train the model. Default is false.
    data_name: str
        For data with binary outcome variable.
        Specify the data source.
        'speed': the speed dating dataset.
        'epi': the epilepsy dataset.
    """
    # select the csv files to be loaded
    if data_type == 'cont':
        if subset is True:
            dgp = random.randint(1, 17)
            file_names_list = file_names_cont_subset(dgp)
        else:
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
    treat = np.array(row[4], dtype=np.int64)
    x = np.array(row[5:], dtype=np.float32)
    return ATE, EY1, EY0, y, treat, x


def row_processor_bin_test(row):
    ATE = np.array(row[0], dtype=np.float32)
    EY1 = np.array(row[1], dtype=np.float32)
    EY0 = np.array(row[2], dtype=np.float32)
    label = np.array(row[3], dtype=np.int64)
    treat = np.array(row[4], dtype=np.int64)
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


# create simulation data
class SimulationData_Cont(Dataset):
    """
    simulation dataset with continuous outcome variable

    seed: float
        random seed
    size: float
        size of the simulation dataset
    """
    def __init__(self, seed, size):
        self.seed = seed
        self.size = size
        self.treat, self.x, self.y = [], [], []

    def generate_data(self):
        sigma = 1.0
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        for i in range(int(self.size)):
            treat_temp = np.random.binomial(1, 0.5)

            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            x_temp = np.repeat(ee, 5)
            for j in range(5):
                x_temp[j] += np.sqrt(sigma) * np.random.normal(0, 1)
            x_temp /= np.sqrt(2)

            y_temp = 5 * x_temp[1] / (1 + x_temp[0] * x_temp[0]) + 5 * np.sin(x_temp[2] * x_temp[3]) + 2 * x_temp[
                4] + np.random.normal(0, 1)
            y_temp = np.reshape(y_temp, 1)

            self.treat.append(treat_temp)
            self.x.append(x_temp.astype('float32'))
            self.y.append(y_temp.astype('float32'))
        return self.treat, self.y, self.x

    def __len__(self):
        return int(self.size)

    def __getitem__(self, idx):
        self.generate_data()
        y = self.y[idx]
        treat = self.treat[idx]
        x = self.x[idx]
        return y, treat, x


class SimulationData_Bin(Dataset):
    """
    simulation dataset with binary outcome variable.

    seed: float
        random seed
    size: float
        size of the simulation dataset
    """
    def __init__(self, seed, size):
        self.seed = seed
        self.size = size
        self.treat, self.x, self.label = [], [], []

    def generate_data(self):
        sigma = 1.0
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        for i in range(int(self.size)):
            treat_temp = np.random.binomial(1, 0.5)

            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            x_temp = np.repeat(ee, 5)
            for j in range(5):
                x_temp[j] += np.sqrt(sigma) * np.random.normal(0, 1)
            x_temp /= np.sqrt(2)

            score = 5 * x_temp[1] / (1 + x_temp[0] * x_temp[0]) + 5 * np.sin(x_temp[2] * x_temp[3]) + 2 * x_temp[
                4] + np.random.normal(0, 1)
            prob = np.exp(score) / (1 + np.exp(score))
            label_temp = np.random.binomial(1, prob)

            self.treat.append(treat_temp)
            self.x.append(x_temp.astype('float32'))
            self.label.append(label_temp)
        return self.treat, self.x, self.label

    def __len__(self):
        return int(self.size)

    def __getitem__(self, idx):
        self.generate_data()
        label = self.label[idx]
        treat = self.treat[idx]
        x = self.x[idx]
        return label, treat, x
