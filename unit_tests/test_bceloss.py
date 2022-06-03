import torch
import torch.nn as nn
from network import Net
from data import SimulationData_Cont
from torch.utils.data import DataLoader
from torch.optim import SGD
import numpy as np

# training parameters
loss_func_sum = nn.MSELoss(reduction='sum')
sse = nn.MSELoss(reduction='sum')
impute_lrs = [1e-7, 1e-7, 1e-7]
ita = 0.5
batch_size = 5
para_lrs = [1e-4]
para_momentum = 0.9

# simulation dataset
train_data = DataLoader(SimulationData_Cont(1, 500), batch_size=batch_size)
dim_data = len(train_data.dataset)

# get input size
train_iter = iter(train_data)
y, treat, x = next(train_iter)
input_dim = x[0].size(dim=0)

# network parameters
num_hidden = 3
hidden_dim = [4, 3, 2]
output_dim = 1
sigma_list = [1e-3, 1e-4, 1e-5, 1e-6]
mh_step = 5
treat_layer = 1
treat_node = 1

net = Net(num_hidden, hidden_dim, input_dim, output_dim, treat_layer, treat_node)
hidden_list = net.backward_imputation(mh_step, impute_lrs, ita, loss_func_sum, sigma_list, x, y, treat)

