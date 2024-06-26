from scipy.stats import truncnorm, bernoulli, beta, norm
from torch.utils.data import Dataset, random_split, ConcatDataset, Subset
from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np
import pandas as pd
import torch
import os


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
    test_set = cross_fit_set.pop(cross_fit_no-1 if cross_fit_no < cross_val else 0)
    train_set = ConcatDataset(cross_fit_set)

    val_indices = val_set.indices
    test_indices = test_set.indices
    train_indices = list(np.concatenate([cross_fit_set[i].indices for i in range(cross_val-2)]).flat)

    x_scalar = StandardScaler()
    y_scalar = StandardScaler()

    if x_scale:
        x_scalar.fit(data.num_var[train_indices])
        data.num_var[train_indices] = np.array(x_scalar.transform(data.num_var[train_indices]))
        data.num_var[val_indices] = np.array(x_scalar.transform(data.num_var[val_indices]))
        data.num_var[test_indices] = np.array(x_scalar.transform(data.num_var[test_indices]))

    if y_scale:
        y_scalar.fit(data.y[train_indices])
        data.y[train_indices] = np.array(y_scalar.transform(data.y[train_indices]))
        data.y[val_indices] = np.array(y_scalar.transform(data.y[val_indices]))
        data.y[test_indices] = np.array(y_scalar.transform(data.y[test_indices]))

    return train_set, val_set, test_set, x_scalar, y_scalar


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
    data.x[data.mis_idx4, 3] = data.x[data.obs_idx4, 3].median()

    # separate missing data and observed data: only the training set will contain missing values
    mis_idx = np.union1d(data.mis_idx1, data.mis_idx4)
    obs_idx = np.intersect1d(data.obs_idx1, data.obs_idx4, assume_unique=True)

    np.random.seed(partition_seed)
    np.random.shuffle(obs_idx)
    val_idx, test_idx, train_obs_idx = np.split(obs_idx, [1000, 2000])
    train_idx = np.concatenate((mis_idx, train_obs_idx))
    train_set, val_set, test_set = Subset(data, train_idx), Subset(data, val_idx), Subset(data, test_idx)
    return train_set, val_set, test_set


class Simulation1_complete(Dataset):
    """
    generate simulation dataset;
    covariates are correlated based on the pre-specified AR(2) process
    seed: int
        seed to control randomness
    """
    def __init__(self, seed):
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

        # generate treatment effect
        y = -4*h31+2*h32
        y_count = -4*h31_count+2*h32_count
        tau = (y-y_count) * (2*treat-1)

        # generate outcome variable
        y = y + np.random.normal(0, 1)

        self.x = torch.FloatTensor(x).to(device)
        self.treat = torch.FloatTensor(treat).to(device)
        self.y = torch.FloatTensor(y).reshape(self.data_size, 1).to(device)
        self.tau = torch.FloatTensor(tau).reshape(self.data_size, 1).to(device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        tau = self.tau[idx]
        return y, treat, x, tau


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


# BRCA dataset
class BRCA(Dataset):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = pd.read_csv("./raw_data/tcga/brca_data.csv")
        num_col = ['years_to_birth', 'date_of_initial_pathologic_diagnosis', 'number_of_lymph_nodes'] + \
                  data.columns[23:].to_list()
        self.data_size = len(data.index)

        self.y = torch.FloatTensor(np.array(data['vital_status'])).long().to(self.device)
        self.treat = torch.FloatTensor(np.array(data['radiation_therapy'], dtype=np.float32)).to(self.device)
        self.num_var = np.array(data[num_col], dtype=np.float32)
        self.cat_var = np.array(data.loc[:, ~data.columns.isin(['vital_status', 'radiation_therapy', 'days_to_death',
                                                                'days_to_last_followup', *num_col])], dtype=np.float32)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        treat = self.treat[idx]
        x = torch.FloatTensor(np.concatenate((self.num_var[idx], self.cat_var[idx]))).to(self.device)
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

        # generate treatment effect
        y = -4*h31+2*h32
        y_count = -4*h31_count+2*h32_count
        tau = (y-y_count) * (2*treat-1)

        # generate outcome variable
        y = y + np.random.normal(0, 1)

        # generate missing values: note that only x1 and x4 has missing values
        if miss_pattern == 'mnar':  # generates observed indicator
            # for x1
            beta1 = np.concatenate((np.array([4, -2]), np.resize([0.1, 0, -0.1, 0], 100)))
            score1 = beta1[0] + beta1[1]*treat + np.matmul(x, beta1[2:])  # R1|X, A
            prob1 = np.exp(score1)/(1+np.exp(score1))
            obs_ind1 = bernoulli.rvs(p=prob1)  # 1 is observed and 0 is missing

            # for x4
            beta4 = np.concatenate((np.array([4, -2]), np.resize([0, -0.1, 0, 0.1], 100)))
            score4 = beta4[0] + beta4[1]*treat + np.matmul(x, beta4[2:])  # R4|X, A
            prob4 = np.exp(score4)/(1+np.exp(score4))
            obs_ind4 = bernoulli.rvs(p=prob4)  # 1 is observed and 0 is missing

        elif miss_pattern == 'mar':  # randomly delete 10% of the data entries in training set
            # for x1
            obs_ind1 = np.array([0]*1000 + [1]*11000)
            np.random.shuffle(obs_ind1)

            # for x4
            obs_ind4 = np.array([0]*1000 + [1]*11000)
            np.random.shuffle(obs_ind4)

        self.obs_idx1 = np.array(np.nonzero(obs_ind1)).flatten()
        self.mis_idx1 = np.array(np.nonzero(1-obs_ind1)).flatten()
        x[self.mis_idx1, 0] = np.nan

        self.obs_idx4 = np.array(np.nonzero(obs_ind4)).flatten()
        self.mis_idx4 = np.array(np.nonzero(1-obs_ind4)).flatten()
        x[self.mis_idx4, 3] = np.nan

        miss_ind = np.stack((1-obs_ind1, 1-obs_ind4), axis=1)

        self.x = torch.FloatTensor(x).to(device)
        self.treat = torch.FloatTensor(treat).to(device)
        self.y = torch.FloatTensor(y).reshape(self.data_size, 1).to(device)
        self.tau = torch.FloatTensor(tau).reshape(self.data_size, 1).to(device)
        self.miss_ind = torch.FloatTensor(miss_ind).to(device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        tau = self.tau[idx]
        miss_ind = self.miss_ind[idx]
        return y, treat, x, miss_ind, tau


class acic_bench(Dataset):
    """
    Load ACIC data for experimental benchmark
    need to scale the numerical variables
    """
    def __init__(self, mode='train'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mode == 'train':
            csv_name = 'high1601.csv'
        elif mode == 'test':
            csv_name = 'highDim_testdataset1.csv'
        csv_dir = os.path.join('./raw_data/acic_bench', csv_name)
        data = pd.read_csv(csv_dir)

        self.data_size = len(data.index)

        # extract column names for binary variables
        bin_col = []
        for col in data.columns:
            if len(data[col].unique()) == 2:
                bin_col.append(col)
        bin_col = bin_col[2:]

        self.y = torch.FloatTensor(np.array(data['Y'], dtype=np.float32)).long().to(self.device)
        self.treat = torch.FloatTensor(np.array(data['A'], dtype=np.float32)).to(self.device)

        # standardize covariates
        bin_var = np.array(data[bin_col], dtype=np.float32)
        num_var = np.array(data.loc[:, ~data.columns.isin(['Y', 'A', *bin_col])], dtype=np.float32)
        x_scalar = StandardScaler()
        x_scalar.fit(num_var)
        num_var = np.array(x_scalar.transform(num_var))
        self.x = torch.FloatTensor(np.concatenate((num_var, bin_var), axis=1)).to(self.device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        treat = self.treat[idx]
        x = self.x[idx]
        return y, treat, x


class Simulation(Dataset):
    """
    generate simulation data with causal relationship
    input_size: int
        dimension of covariates
    sample_size: int
        sample size of the training set
    seed: int
        random seed to generate the dataset
    """
    def __init__(self, input_size, sample_size, seed):
        self.sample_size = sample_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        one_count, zero_count = 0, 0  # count of the samples in treatment group and control group, respectively
        one_treat, one_x = ([] for _ in range(2))
        zero_treat, zero_x = ([] for _ in range(2))

        np.random.seed(seed)
        torch.manual_seed(seed)
        while min(one_count, zero_count) < self.sample_size // 2:
            # generate covariates
            ee = truncnorm.rvs(-10, 10)
            x_temp = truncnorm.rvs(-10, 10, size=input_size) + ee
            x_temp /= np.sqrt(2)

            # generate treatment
            temp = (norm.cdf(x_temp[0]) + norm.cdf(x_temp[2]) + norm.cdf(x_temp[4]))/3
            temp = beta.cdf(temp, 2, 4)
            prop = 0.25 * (1+temp)
            treat_temp = bernoulli.rvs(p=prop)

            if treat_temp == 1:
                one_count += 1
                one_x.append(x_temp)
                one_treat.append(treat_temp)
            else:
                zero_count += 1
                zero_x.append(x_temp)
                zero_treat.append(treat_temp)

        x = np.array(one_x[:(self.sample_size // 2)] + zero_x[:(self.sample_size // 2)])
        treat = np.array(one_treat[:(self.sample_size // 2)] + zero_treat[:(self.sample_size // 2)])

        # generate outcome
        c = 5*x[..., 2]/(1+x[..., 3]**2) + 2*x[..., 4]
        f1 = 2/(1+np.exp(-x[..., 0]+0.5))
        f2 = 2/(1+np.exp(-x[..., 1]+0.5))
        ita = f1*f2 - f1*f2.mean()
        y = c + (3+ita)*treat + 0.25*norm.rvs(size=self.sample_size)

        self.x = torch.FloatTensor(x).to(self.device)
        self.y = torch.FloatTensor(y.reshape(self.sample_size, 1)).to(self.device)
        self.treat = torch.FloatTensor(treat).to(self.device)

    def __len__(self):
        return int(self.sample_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        treat = self.treat[idx]
        x = self.x[idx]
        return y, treat, x


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
        one_treat, one_x, one_y, one_tau_count = ([] for _ in range(4))
        zero_treat, zero_x, zero_y, zero_tau_count = ([] for _ in range(4))

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

            # generate true treatment effect
            y_temp = -4*h31+2*h32
            y_count_temp = -4*h31_count+2*h32_count
            tau = (y_temp-y_count_temp) *(2*treat_temp-1)

            # generate outcome
            y_temp = y_temp + np.random.normal(0, 1)

            if treat_temp == 1:
                one_count += 1
                one_x.append(x_temp)
                one_y.append(y_temp)
                one_treat.append(treat_temp)
                one_tau_count.append(tau)
            else:
                zero_count += 1
                zero_x.append(x_temp)
                zero_y.append(y_temp)
                zero_treat.append(treat_temp)
                zero_tau_count.append(tau)

        self.x = torch.FloatTensor(np.array(one_x[:(data_size // 2)] + zero_x[:(data_size // 2)])).to(device)
        self.y = torch.FloatTensor(np.array(one_y[:(data_size // 2)] + zero_y[:(data_size // 2)]).
                                   reshape(self.data_size, 1)).to(device)
        self.treat = torch.FloatTensor(np.array(one_treat[:(data_size // 2)] + zero_treat[:(data_size // 2)])).to(device)
        self.tau = torch.FloatTensor(np.array(one_tau_count[:(data_size // 2)] + zero_tau_count[:(data_size // 2)]).
                                     reshape(self.data_size, 1)).to(device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        tau = self.tau[idx]
        return y, treat, x, tau
