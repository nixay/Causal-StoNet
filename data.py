from scipy.stats import truncnorm, bernoulli
from torch.utils.data import Dataset, random_split, ConcatDataset
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
import torch
import os


def data_preprocess(data, partition_seed, cross_fit_no, scale=True):
    """
    data: Dataset object
        map-style dataset (only map-style dataset has __len__() property)
    partition_seed: int
        seed to randomly partition the dataset into train set and validation set
    cross_fit_no: int
        the data subset that is going to be used as train set; note that we use three-fold cross fitting.
    scale: bool
        whether the input and the output will be scaled
    """
    data_size = data.__len__()
    size = int(data_size/3)
    cross_fit_set = random_split(data, [size, size, data_size-2*size],
                                                generator=torch.Generator().manual_seed(partition_seed))
    val_set = cross_fit_set.pop(cross_fit_no-1)
    train_set = ConcatDataset(cross_fit_set)

    val_indices = val_set.indices
    train_indices = cross_fit_set[0].indices + cross_fit_set[1].indices

    if scale:
        x_scalar = RobustScaler()
        x_scalar.fit(data.num_var[train_indices])
        data.num_var[train_indices] = np.array(x_scalar.transform(data.num_var[train_indices]))
        data.num_var[val_indices] = np.array(x_scalar.transform(data.num_var[val_indices]))

        y_scalar = RobustScaler()
        y_scalar.fit(data.y[train_indices])
        data.y[train_indices] = np.array(y_scalar.transform(data.y[train_indices]))
        data.y[val_indices] = np.array(y_scalar.transform(data.y[val_indices]))

        return train_set, val_set, x_scalar, y_scalar
    else:
        return train_set, val_set


# Simulation Dataset
class SimData_Causal(Dataset):
    """
    generate simulation data with causal relationship; inputs are correlated; there is overlap between inputs for
    outcome variable and inputs for treatment. Note that when using this dataset, 'shuffle' argument has to be set
    to be true in dataloader.
    input_size: int
        the dimension of input variable
    seed: int
        random seed to generate the dataset
    data_size: int
        sample size of the dataset
    """
    def __init__(self, input_size, seed, data_size):
        self.data_size = data_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        one_count, zero_count = 0, 0  # count of the samples in treatment group and control group, respectively
        one_treat, one_x, one_y, one_y_count = ([] for _ in range(4))
        zero_treat, zero_x, zero_y, zero_y_count = ([] for _ in range(4))

        np.random.seed(seed)
        torch.manual_seed(seed)
        while min(one_count, zero_count) < data_size // 2:
            # generate x
            ee = truncnorm.rvs(-10, 10)
            x_temp = truncnorm.rvs(-10, 10, size=input_size) + ee
            x_temp /= np.sqrt(2)

            # nodes in the first hidden layer
            h11 = np.tanh(2*x_temp[0]-4*x_temp[3])
            h12 = np.tanh(-4*x_temp[0]+2*x_temp[4])
            h13 = np.tanh(4*x_temp[1]+3*x_temp[2])
            h14 = np.tanh(x_temp[3]-3*x_temp[4])

            # nodes in the second hidden layer
            h21 = np.tanh(-4*h11+h13)
            h22 = np.tanh(h12-h13)
            h23 = np.tanh(h13-2*h14)

            # generate treatment
            prob = np.exp(h22)/(1 + np.exp(h22))
            treat_temp = bernoulli.rvs(p=prob)

            # nodes in the third hidden layer
            h31 = np.tanh(3*h21-4*treat_temp)
            h32 = np.tanh(2*treat_temp-h23)

            # counterfactual nodes in the third hidden layer
            h31_count = np.tanh(3*h21-4*(1-treat_temp))
            h32_count = np.tanh(2*(1-treat_temp)-h23)

            # generate outcome variable
            y_temp = np.tanh(-3*h31+4*h32) + np.random.normal(0, 1)

            # generate counterfactual outcome variable
            y_count_temp = np.tanh(-3*h31_count+4*h32_count) + np.random.normal(0, 1)

            if treat_temp == 1:
                one_count += 1
                one_x.append(x_temp)
                one_y.append(y_temp)
                one_treat.append(treat_temp)
                one_y_count.append(y_count_temp)
            else:
                zero_count += 1
                zero_x.append(x_temp)
                zero_y.append(y_temp)
                zero_treat.append(treat_temp)
                zero_y_count.append(y_count_temp)

        self.x = torch.FloatTensor(np.array(one_x[:(data_size // 2)] + zero_x[:(data_size // 2)])).to(device)
        self.y = torch.FloatTensor(np.array(one_y[:(data_size // 2)] + zero_y[:(data_size // 2)]).
                                   reshape(self.data_size, 1)).to(device)
        self.treat = torch.FloatTensor(np.array(one_treat[:(data_size // 2)] + zero_treat[:(data_size // 2)])).to(device)
        self.y_count = torch.FloatTensor(np.array(one_y_count[:(data_size // 2)] + zero_y_count[:(data_size // 2)]).
                                         reshape(self.data_size, 1)).to(device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        return y, treat, x

    def true_ate(self):
        ate = torch.mean(torch.flatten((self.y - self.y_count)) * (2*self.treat-1))
        return ate


class SimData_Causal_Ind(Dataset):
    """
    generate simulation data with causal relationship; inputs are independent; there is no overlap between inputs for
    outcome variable and inputs for treatment. Note that when using this dataset, 'shuffle' argument has to be set
    to be true in dataloader.
    input_size: int
        the dimension of input variable
    seed: int
        random seed to generate the dataset
    data_size: int
        sample size of the dataset
    """
    def __init__(self, input_size, seed, data_size):
        self.data_size = data_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        one_count, zero_count = 0, 0  # count of the data points with treatment 1 and treatment 0
        one_treat, one_x, one_y, one_y_count = ([] for _ in range(4))
        zero_treat, zero_x, zero_y, zero_y_count = ([] for _ in range(4))

        np.random.seed(seed)
        torch.manual_seed(seed)
        while min(one_count, zero_count) < data_size // 2:
            # generate x
            x_temp = truncnorm.rvs(-10, 10, size=input_size)

            # nodes in the first hidden layer
            h11 = np.tanh(2*x_temp[0]-4*x_temp[2])
            h12 = np.tanh(-4*x_temp[1]+2*x_temp[3])
            h13 = np.tanh(4*x_temp[3]+3*x_temp[1])
            h14 = np.tanh(x_temp[2]-3*x_temp[4])

            # nodes in the second hidden layer
            h21 = np.tanh(-4*h11+h14)
            h22 = np.tanh(h12-h13)
            h23 = np.tanh(h11-2*h14)

            # generate treatment
            prob = np.exp(h22)/(1 + np.exp(h22))
            treat_temp = bernoulli.rvs(p=prob)

            # nodes in the third hidden layer
            h31 = np.tanh(3*h21-4*treat_temp)
            h32 = np.tanh(2*treat_temp-h23)

            # counterfactual nodes in the third hidden layer
            h31_count = np.tanh(3*h21-4*(1-treat_temp))
            h32_count = np.tanh(2*(1-treat_temp)-h23)

            # generate outcome variable
            y_temp = np.tanh(-3*h31+4*h32) + np.random.normal(0, 1)

            # generate counterfactual outcome variable
            y_count_temp = np.tanh(-3*h31_count+4*h32_count) + np.random.normal(0, 1)

            if treat_temp == 1:
                one_count += 1
                one_x.append(x_temp)
                one_y.append(y_temp)
                one_treat.append(treat_temp)
                one_y_count.append(y_count_temp)
            else:
                zero_count += 1
                zero_x.append(x_temp)
                zero_y.append(y_temp)
                zero_treat.append(treat_temp)
                zero_y_count.append(y_count_temp)

        self.x = torch.FloatTensor(np.array(one_x[:(data_size // 2)] + zero_x[:(data_size // 2)])).to(device)
        self.y = torch.FloatTensor(np.array(one_y[:(data_size // 2)] + zero_y[:(data_size // 2)]).
                                   reshape(self.data_size, 1)).to(device)
        self.treat = torch.FloatTensor(np.array(one_treat[:(data_size // 2)] + zero_treat[:(data_size // 2)])).to(device)
        self.y_count = torch.FloatTensor(np.array(one_y_count[:(data_size // 2)] + zero_y_count[:(data_size // 2)]).
                                         reshape(self.data_size, 1)).to(device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        return y, treat, x

    def true_ate(self):
        ate = torch.mean(torch.flatten((self.y - self.y_count)) * (2*self.treat-1))
        return ate


# ACIC Dataset with Continuous Variable and Homogeneous Treatment Effect
class acic_data_homo(Dataset):
    """
    Load ACIC data with specific data generating process number
    need to scale the numerical variables
    dgp: int
        data generating process number
    """
    def __init__(self, dgp):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # extract csv file names under the specified dgp
        file_names_temp = pd.read_excel('./raw_data/acic/DatasetsCorrespondence.xlsx', header=None)
        file_index = file_names_temp[0].str.contains("CHDScenario"+str(dgp)+"DS", case=True, regex=False)
        file_names_list = list(file_names_temp[1][file_index][:25])

        # read and concatenate the csv files
        data = []
        root_dir = './raw_data/acic/data_subset'
        for file_name in file_names_list:
            dir = os.path.join(root_dir, file_name + ".CSV")
            file = pd.read_csv(dir)
            data.append(file)
        data = pd.concat(data, ignore_index=True)
        self.data_size = len(data.index)

        # extract column names for categorical variables
        cat_col = []
        for col in data.columns:
            if len(data[col].unique()) <= data[col].max() + 1:
                cat_col.append(col)
        cat_col = cat_col[1:]

        self.y = np.array(data['Y'], dtype=np.float32).reshape(self.data_size, 1)
        self.treat = torch.FloatTensor(np.array(data['A'], dtype=np.float32)).to(self.device)
        self.cat_var = np.array(data[cat_col], dtype=np.float32)
        self.num_var = np.array(data.loc[:, ~data.columns.isin(['Y', 'A', *cat_col])],
                            dtype=np.float32)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = torch.FloatTensor(self.y[idx]).to(self.device)
        treat = self.treat[idx]
        x = torch.FloatTensor(np.concatenate((self.num_var[idx], self.cat_var[idx]))).to(self.device)
        return y, treat, x


# ACIC Dataset with Continuous Variable and Heterogeneous Treatment Effect
class acic_data_hete(Dataset):
    """
    load ACIC test data that combines different dgp to create heterogeneous treatment effect
    need to scale the numerical variables
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # read and concatenate the csv files
        data = []
        file_names_list = ['cont2', 'cont3', 'cont6', 'cont7']
        root_dir = './raw_data/acic/test_data'
        for file_name in file_names_list:
            dir = os.path.join(root_dir, file_name + ".csv")
            file = pd.read_csv(dir)
            data.append(file)
        data = pd.concat(data, ignore_index=True)
        self.data_size = len(data.index)

        # extract column names for categorical variables
        # note that for this dataset, the extracted cat_cols may not be real categorical variables
        # for some extracted columns, the values don't seem to be encoding of categorical variables
        cat_col = []
        for col in data.columns:
            if len(data[col].unique()) == data[col].max() + 1:
                cat_col.append(col)
        cat_col = cat_col[1:]

        self.ate = torch.FloatTensor(np.array(data['ATE'], dtype=np.float32)).to(self.device)
        self.y1 = torch.FloatTensor(np.array(data['EY1'], dtype=np.float32)).to(self.device)
        self.y0 = torch.FloatTensor(np.array(data['EY0'], dtype=np.float32)).to(self.device)

        self.y = np.array(data['Y'], dtype=np.float32).reshape(self.data_size, 1)
        self.treat = torch.FloatTensor(np.array(data['A'], dtype=np.float32)).to(self.device)
        self.cat_var = np.array(data[cat_col], dtype=np.float32)
        self.num_var = np.array(data.loc[:, ~data.columns.isin(['ATE', 'EY1', 'EY0', 'Y', 'A', *cat_col])],
                                            dtype=np.float32)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = torch.FloatTensor(self.y[idx]).to(self.device)
        treat = self.treat[idx]
        x = torch.FloatTensor(np.concatenate((self.num_var[idx], self.cat_var[idx]))).to(self.device)
        return y, treat, x


# 401k Dataset
class PensionData(Dataset):
    """
    load 401k dataset
    need to scale the numerical variables
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = pd.read_csv("./raw_data/401k/401k.csv")
        ####################################################
        # drop_ind = data.loc[data['net_tfa'] >= 0.5e6].index
        # data = data.drop(drop_ind)
        #####################################################
        cat_col = ['db', 'marr', 'male', 'twoearn', 'pira', 'nohs', 'hs', 'smcol', 'col', 'hown']
        self.data_size = len(data.index)

        self.y = np.array(data['net_tfa'], dtype=np.float32).reshape(self.data_size, 1)
        self.treat = torch.FloatTensor(np.array(data['e401'], dtype=np.float32)).to(self.device)
        self.cat_var = np.array(data[cat_col], dtype=np.float32)
        self.num_var = np.array(data.loc[:, ~data.columns.isin(['net_tfa', 'e401', *cat_col])],
                                                  dtype=np.float32)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = torch.FloatTensor(self.y[idx]).to(self.device)
        treat = self.treat[idx]
        x = torch.FloatTensor(np.concatenate((self.num_var[idx], self.cat_var[idx]))).to(self.device)
        return y, treat, x


# Twins Dataset
class TwinsData(Dataset):
    """
    load twins dataset
    all the columns are categorical (or binary) variables, no need to scale the data
    """
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = pd.read_csv("./raw_data/twins/twins_data.csv")
        self.data_size = len(data.index)

        self.y = torch.FloatTensor(np.array(data['y'])).long().to(device)
        self.treat = torch.FloatTensor(np.array(data['treat'], dtype=np.float32)).to(device)
        self.counter = torch.FloatTensor(np.array(data['counter'], dtype=np.float32)).to(device)
        self.x = torch.FloatTensor(np.array(data.loc[:, ~data.columns.isin(['y', 'treat', 'counter'])], dtype=np.float32)).to(device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        treat = self.treat[idx]
        x = self.x[idx]
        return y, treat, x
