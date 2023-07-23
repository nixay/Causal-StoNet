from scipy.stats import truncnorm, bernoulli
from torch.utils.data import Dataset, random_split, ConcatDataset, Subset
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
import torch
import os


def true_ate(y, treat, y_count):
    ate = torch.mean(torch.flatten((y - y_count)) * (2*treat-1))
    return ate


def prune_threshold(sigma_0, sigma_1, lambda_n):
    threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(sigma_1 / sigma_0)) / (
            0.5 / sigma_0 - 0.5 / sigma_1))
    return threshold


def data_preprocess(data, partition_seed, cross_fit_no, cross_val=3, x_scale=True, y_scale=True):
    """
    data: Dataset object
        map-style dataset (only map-style dataset has __len__() property)
    partition_seed: int
        seed to randomly partition the dataset into train set and validation set
    cross_fit_no: int
        the data subset that is going to be used as train set; note that we use three-fold cross fitting.
    cross_val: int
        the number of cross-validation folds
    x_scale: bool
        whether the input variables will be scaled
    y_scale: bool
        whether the output variable will be scaled
    """
    data_size = data.__len__()
    size = int(data_size/cross_val)
    size_list = [size]*(cross_val-1)
    size_list.append(data_size-(cross_val-1)*size)
    cross_fit_set = random_split(data, size_list, generator=torch.Generator().manual_seed(partition_seed))
    val_set = cross_fit_set.pop(cross_fit_no-1)
    train_set = ConcatDataset(cross_fit_set)

    val_indices = val_set.indices
    train_indices = list(np.concatenate([cross_fit_set[i].indices for i in range(cross_val-1)]).flat)

    x_scalar = RobustScaler()
    y_scalar = RobustScaler()

    if x_scale:
        x_scalar.fit(data.num_var[train_indices])
        data.num_var[train_indices] = np.array(x_scalar.transform(data.num_var[train_indices]))
        data.num_var[val_indices] = np.array(x_scalar.transform(data.num_var[val_indices]))

    if y_scale:
        y_scalar.fit(data.y[train_indices])
        data.y[train_indices] = np.array(y_scalar.transform(data.y[train_indices]))
        data.y[val_indices] = np.array(y_scalar.transform(data.y[val_indices]))

    return train_set, val_set, x_scalar, y_scalar


def miss_simulated_preprocess(data, partition_seed):
    """
    preprocess the simulated dataset with missing values.
    data: Dataset object
        output of SimData_Missing
    seed: int
        random seed to control the partition of the dataset
    """
    # fill the nan values by the median of observed values
    data.x[data.mis_idx1, 0] = data.x[data.obs_idx1, 0].median()
    # data.x[data.mis_idx4, 3] = data.x[data.obs_idx4, 3].median()
    data.x[data.mis_idx5, 4] = data.x[data.obs_idx5, 4].median()

    # separate missing data and observed data: only the training set will contain missing values
    # mis_idx = np.union1d(data.mis_idx1, data.mis_idx4)
    # obs_idx = np.intersect1d(data.obs_idx1, data.obs_idx4, assume_unique=True)
    mis_idx = np.union1d(data.mis_idx1, data.mis_idx5)
    obs_idx = np.intersect1d(data.obs_idx1, data.obs_idx5, assume_unique=True)

    np.random.seed(partition_seed)
    np.random.shuffle(obs_idx)
    val_idx, test_idx, train_obs_idx = np.split(obs_idx, [1000, 2000])
    train_idx = np.concatenate((mis_idx, train_obs_idx))
    train_set, val_set, test_set = Subset(data, train_idx), Subset(data, val_idx), Subset(data, test_idx)
    return train_set, val_set, test_set


# Simulation Dataset
class SimData_Causal(Dataset):
    """
    generate simulation data with causal relationship
    Note that when using this dataset, 'shuffle' argument has to be set to be true in dataloader.
    input_size: int
        the dimension of input variable
    seed: int
        random seed to generate the dataset
    data_size: int
        sample size of the dataset
    cor: boolean
        if True: inputs are correlated
        if False: inputs are independent
        The default is true

    """
    def __init__(self, input_size, seed, data_size, cor=True):
        self.data_size = data_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        one_count, zero_count = 0, 0  # count of the samples in treatment group and control group, respectively
        one_treat, one_x, one_y, one_y_count = ([] for _ in range(4))
        zero_treat, zero_x, zero_y, zero_y_count = ([] for _ in range(4))

        np.random.seed(seed)
        torch.manual_seed(seed)
        while min(one_count, zero_count) < data_size // 2:
            # generate x
            if cor:
                ee = truncnorm.rvs(-10, 10)
                x_temp = truncnorm.rvs(-10, 10, size=input_size) + ee
                x_temp /= np.sqrt(2)
            else:
                x_temp = truncnorm.rvs(-10, 10, size=input_size)

            # nodes in the first hidden layer
            h11 = np.tanh(2*x_temp[0]+1*x_temp[3])
            h12 = np.tanh(-x_temp[0]-2*x_temp[4])
            h13 = np.tanh(2*x_temp[1]-2*x_temp[2])
            h14 = np.tanh(-2*x_temp[3]+1*x_temp[4])

            # nodes in the second hidden layer
            h21 = np.tanh(-2*h11+h13)
            h22 = h12-h13
            h23 = np.tanh(h13-2*h14)

            # generate treatment
            prob = np.exp(h22)/(1 + np.exp(h22))
            treat_temp = bernoulli.rvs(p=prob)

            # nodes in the third hidden layer
            h31 = np.tanh(1*h21-2*treat_temp)
            h32 = np.tanh(-1*treat_temp+2*h23)

            # counterfactual nodes in the third hidden layer
            h31_count = np.tanh(1*h21-2*(1-treat_temp))
            h32_count = np.tanh(-1*(1-treat_temp)+2*h23)

            # generate outcome variable
            y_temp = -4*h31+2*h32 + np.random.normal(0, 1)

            # generate counterfactual outcome variable
            y_count_temp = -4*h31_count+2*h32_count + np.random.normal(0, 1)

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
        y_count = self.y_count[idx]
        return y, treat, x, y_count


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

        csv_name = 'acic_homo' + str(dgp) + '.csv'
        csv_dir = os.path.join('./raw_data/acic', csv_name)
        data = pd.read_csv(csv_dir)

        self.data_size = len(data.index)

        # extract column names for categorical variables
        cat_col = []
        for col in data.columns:
            if data[col].abs().max() <= 10:
                if len(data[col].unique()) <= data[col].max() + 1:
                    cat_col.append(col)
        cat_col = cat_col[1:]

        self.y = np.array(data['Y'], dtype=np.float32).reshape(self.data_size, 1)
        self.treat = torch.FloatTensor(np.array(data['A'], dtype=np.float32)).to(self.device)
        self.cat_var = np.array(data[cat_col], dtype=np.float32)
        self.num_var = np.array(data.loc[:, ~data.columns.isin(['Y', 'A', *cat_col])], dtype=np.float32)

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
        cat_col = []
        for col in data.columns:
            if data[col].abs().max() <= 10:
                if len(data[col].unique()) <= data[col].max() + 1:
                    cat_col.append(col)
        cat_col = cat_col[1:]

        self.ate = torch.FloatTensor(np.array(data['ATE'], dtype=np.float32)).to(self.device)
        self.y1 = torch.FloatTensor(np.array(data['EY1'], dtype=np.float32)).to(self.device)
        self.y0 = torch.FloatTensor(np.array(data['EY0'], dtype=np.float32)).to(self.device)

        self.y = np.array(data['Y'], dtype=np.float32).reshape(self.data_size, 1)
        self.treat = torch.FloatTensor(np.array(data['A'], dtype=np.float32)).to(self.device)
        self.cat_var = np.array(data[cat_col], dtype=np.float32)
        self.num_var = np.array(data.loc[:, ~data.columns.isin(['ATE', 'EY1', 'EY0', 'Y', 'A', *cat_col])], dtype=np.float32)

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
        self.num_var = np.array(data.loc[:, ~data.columns.isin(['net_tfa', 'e401', *cat_col])], dtype=np.float32)

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


# TCGA Dataset
class TCGA(Dataset):
    # note that for multi-level treatment dataset, the treatment variable has to be modeled as [treat1, treat2, control]
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = pd.read_csv("./raw_data/tcga/tcga_data_screened.csv")
        self.data_size = len(data.index)

        self.treat = torch.FloatTensor(np.array(data[['rad', 'sur', 'control']], dtype=np.float32)).to(self.device)
        self.y = torch.FloatTensor(np.array(data['recur'])).long().to(self.device)
        self.num_var = np.array(data.loc[:, ~data.columns.isin(['recur', 'sur', 'rad', 'control'])],
                                dtype=np.float32)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        treat = self.treat[idx]
        x = torch.FloatTensor(self.num_var[idx]).to(self.device)
        return y, treat, x


# Simulation dataset with missing values
class SimData_Missing(Dataset):
    """
    generate simulation dataset with missing values under different missing patterns;
    covariates are correlated based on the pre-specified AR(2) process
    seed: int
        seed to control randomness
    miss_pattern: str
        'mar': missing at random
        'mnar': missing not at random
    """
    def __init__(self, seed, miss_pattern):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load covariates
        csv_name = 'covariates_' + str(seed) + '.csv'
        x = np.loadtxt(os.path.join('./raw_data/sim_missing', csv_name), delimiter=",", skiprows=1)
        self.data_size = x.__len__()

        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # nodes in the first hidden layer
        h11 = np.tanh(2*x[:, 0]+1*x[:, 3])
        h12 = np.tanh(-x[:, 0]-2*x[:, 4])
        h13 = np.tanh(2*x[:, 1]-2*x[:, 2])
        h14 = np.tanh(-2*x[:, 3]+1*x[:, 4])

        # nodes in the second hidden layer
        h21 = np.tanh(-2*h11+h13)
        h22 = h12-h13
        h23 = np.tanh(h13-2*h14)

        # generate treatment
        prob = np.exp(h22)/(1 + np.exp(h22))
        treat = bernoulli.rvs(p=prob)

        # nodes in the third hidden layer
        h31 = np.tanh(1*h21-2*treat)
        h32 = np.tanh(-1*treat+2*h23)

        # counterfactual nodes in the third hidden layer
        h31_count = np.tanh(1*h21-2*(1-treat))
        h32_count = np.tanh(-1*(1-treat)+2*h23)

        # generate outcome variable
        y = -4*h31+2*h32 + np.random.normal(0, 1)

        # generate counterfactual outcome variable
        y_count = -4*h31_count+2*h32_count + np.random.normal(0, 1)

        # generate missing values: note that only x1 and x4 (or x5) has missing values
        if miss_pattern == 'mnar':  # generates observed indicator
            # for x1
            beta1 = np.concatenate((np.array([4, -2]), np.resize([0.1, 0, -0.1, 0], 100)))
            score1 = beta1[0] + beta1[1]*treat + np.matmul(x, beta1[2:])  # R1|X, A
            prob1 = np.exp(score1)/(1+np.exp(score1))
            obs_ind1 = bernoulli.rvs(p=prob1)  # 1 is observed and 0 is missing

            # # for x4
            # beta4 = np.concatenate((np.array([4, -2]), np.resize([0, -0.1, 0, 0.1], 100)))
            # score4 = beta4[0] + beta4[1]*treat + np.matmul(x, beta4[2:])  # R4|X, A
            # prob4 = np.exp(score4)/(1+np.exp(score4))
            # obs_ind4 = bernoulli.rvs(p=prob4)  # 1 is observed and 0 is missing

            # for x5
            beta5 = np.concatenate((np.array([4, -2]), np.resize([0, -0.1, 0, 0.1], 100)))
            score5 = beta5[0] + beta5[1]*treat + np.matmul(x, beta5[2:])  # R4|X, A
            prob5 = np.exp(score5)/(1+np.exp(score5))
            obs_ind5 = bernoulli.rvs(p=prob5)  # 1 is observed and 0 is missing

        elif miss_pattern == 'mar':  # randomly delete 10% of the data entries in training set
            # for x1
            obs_ind1 = np.array([0]*1000 + [1]*11000)
            np.random.shuffle(obs_ind1)

            # # for x4
            # obs_ind4 = np.array([0]*1000 + [1]*11000)
            # np.random.shuffle(obs_ind4)

            # for x5
            obs_ind5 = np.array([0]*1000 + [1]*11000)
            np.random.shuffle(obs_ind5)

        self.obs_idx1 = np.array(np.nonzero(obs_ind1)).flatten()
        self.mis_idx1 = np.array(np.nonzero(1-obs_ind1)).flatten()
        x[self.mis_idx1, 0] = np.nan

        # self.obs_idx4 = np.array(np.nonzero(obs_ind4)).flatten()
        # self.mis_idx4 = np.array(np.nonzero(1-obs_ind4)).flatten()
        # x[self.mis_idx4, 3] = np.nan

        self.obs_idx5 = np.array(np.nonzero(obs_ind5)).flatten()
        self.mis_idx5 = np.array(np.nonzero(1 - obs_ind5)).flatten()
        x[self.mis_idx5, 4] = np.nan

        miss_ind = np.stack((1-obs_ind1, 1-obs_ind5), axis=1)

        self.x = torch.FloatTensor(x).to(device)
        self.treat = torch.FloatTensor(treat).to(device)
        self.y = torch.FloatTensor(y).reshape(self.data_size, 1).to(device)
        self.y_count = torch.FloatTensor(y_count).reshape(self.data_size, 1).to(device)
        self.miss_ind = torch.FloatTensor(miss_ind).to(device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        y_count = self.y_count[idx]
        miss_ind = self.miss_ind[idx]
        return y, treat, x, miss_ind, y_count


# class SimData_Missing_test(Dataset):
#     def __init__(self, seed):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # load covariates
#         csv_name = 'covariates_' + str(seed) + '.csv'
#         x = np.loadtxt(os.path.join('./raw_data/sim_missing', csv_name), delimiter=",", skiprows=1)
#         self.data_size = x.__len__()
#
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         # nodes in the first hidden layer
#         h11 = np.tanh(2*x[:, 0]+1*x[:, 3])
#         h12 = np.tanh(-x[:, 0]-2*x[:, 4])
#         h13 = np.tanh(2*x[:, 1]-2*x[:, 2])
#         h14 = np.tanh(-2*x[:, 3]+1*x[:, 4])
#
#         # nodes in the second hidden layer
#         h21 = np.tanh(-2*h11+h13)
#         h22 = h12-h13
#         h23 = np.tanh(h13-2*h14)
#
#         # generate treatment
#         prob = np.exp(h22)/(1 + np.exp(h22))
#         treat = bernoulli.rvs(p=prob)
#
#         # nodes in the third hidden layer
#         h31 = np.tanh(1*h21-2*treat)
#         h32 = np.tanh(-1*treat+2*h23)
#
#         # counterfactual nodes in the third hidden layer
#         h31_count = np.tanh(1*h21-2*(1-treat))
#         h32_count = np.tanh(-1*(1-treat)+2*h23)
#
#         # generate outcome variable
#         y = -4*h31+2*h32 + np.random.normal(0, 1)
#
#         # generate counterfactual outcome variable
#         y_count = -4*h31_count+2*h32_count + np.random.normal(0, 1)
#
#         self.x = torch.FloatTensor(x).to(self.device)
#         self.treat = torch.FloatTensor(treat).to(self.device)
#         self.y = torch.FloatTensor(y).reshape(self.data_size, 1).to(self.device)
#         self.y_count = torch.FloatTensor(y_count).reshape(self.data_size, 1).to(self.device)
#
#     def __len__(self):
#         return int(self.data_size)
#
#     def __getitem__(self, idx):
#         y = self.y[idx]
#         x = self.x[idx]
#         treat = self.treat[idx]
#         y_count = self.y_count[idx]
#         return y, treat, x, y_count
