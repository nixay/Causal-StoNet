from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.stats import truncnorm


class SimData_Causal(Dataset):
    """
    generate simulation data with causal relationship
    seed: random seed to generate the dataset
    data_size: the number of data points in the dataset
    """
    def __init__(self, input_size, seed, data_size):
        self.data_size = data_size
        self.treat = np.zeros(self.data_size)
        self.x = np.zeros([self.data_size] + [input_size])
        self.y = np.zeros((self.data_size, 1))
        self.y_count = np.zeros((self.data_size, 1))  # counterfactual outcome variable

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        np.random.seed(seed)
        for i in range(self.data_size):
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
            if prob > 0.5:
                treat_temp = 1
            else:
                treat_temp = 0

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

            self.x[i:] = x_temp
            self.treat[i] = treat_temp
            self.y[i] = y_temp
            self.y_count[i] = y_count_temp

        self.x = torch.FloatTensor(self.x).to(device)
        self.treat = torch.FloatTensor(self.treat).to(device)
        self.y = torch.FloatTensor(self.y).to(device)
        self.y_count = torch.FloatTensor(self.y_count).to(device)

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
