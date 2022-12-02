from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.stats import truncnorm, bernoulli


class SimData_Causal(Dataset):
    """
    generate simulation data with causal relationship; inputs are correlated; there is overlap between inputs for
    outcome variable and inputs for treatment. Note that when using this dataset, 'shuffle' argument has to be set
    to be true in dataloader.
    input_size: the number of input variables
    seed: random seed to generate the dataset
    data_size: the number of data points in the dataset
    """
    def __init__(self, input_size, seed, data_size):
        self.data_size = data_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        np.random.seed(seed)

        one_count, zero_count = 0, 0  # count of the samples in treatment group and control group, respectively
        one_treat, one_x, one_y, one_y_count = ([] for _ in range(4))
        zero_treat, zero_x, zero_y, zero_y_count = ([] for _ in range(4))

        while min(one_count, zero_count) < data_size // 2:
            # generate x
            ee = truncnorm.rvs(-10, 10)
            x_temp = truncnorm.rvs(-10, 10, size=input_size) + ee
            x_temp /= np.sqrt(2)

            # nodes in the first hidden layer
            h11 = np.tanh(2*x_temp[0]-4*x_temp[3])
            h12 = np.tanh(-4*x_temp[0]+2*x_temp[4])
            h13 = np.tanh(4*x_temp[1]+3*x_temp[2])
            h14 = np.tanh(4*x_temp[3]-3*x_temp[4])

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
    input_size: the number of input variables
    seed: random seed to generate the dataset
    data_size: the number of data points in the dataset
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
