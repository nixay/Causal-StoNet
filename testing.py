import torch
import torch.nn as nn
from network import Net
from data import acic_data_test
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description='Testing the performance of trained stonet with treatment layer')
parser.add_argument('--seed', default=1, type=int, help='set seed')
parser.add_argument('--regression_flag', default=True, type=int,
                    help='true for regression and false for classification')
parser.add_argument('--layer', default=2, type=int, help='number of hidden layer (not including the treatment layer')
parser.add_argument('--unit', default=[20, 15], type=int, nargs='+', help='number of hidden unit in each layer')
parser.add_argument('--sigma', default=[0.001, 0.0001, 0.00001], type=float, nargs='+',
                    help='variance of each layer for the model')
parser.add_argument('--depth', default=1, type=int, help='number of layers before the treatment layer')
parser.add_argument('--batch_size', default=50, type=int, help='batch size for training')
parser.add_argument('--path', type=str, help='folder that saves the model parameters')

args = parser.parse_args()

seed = args.seed
regression_flag = args.regression_flag
num_hidden = args.layer
hidden_dim = args.unit
sigma_list = args.sigma
treat_depth = args.depth
batch_size = args.batch_size
PATH = args.path

# load data
if regression_flag is True:
    test_set = to_map_style_dataset(acic_data_test('cont'))
else:
    test_set = to_map_style_dataset(acic_data_test('bin'))
test_data = DataLoader(test_set, batch_size=batch_size)

# get input dim and output dim
_, _, x_temp = next(iter(test_data))
input_dim = x_temp[0].size(dim=0)

# settings for loss functions
if regression_flag:
    loss = nn.MSELoss()
    output_dim = 1
else:
    loss = nn.CrossEntropyLoss()
    output_dim = 2  # for binary classification; change accordingly for multi-level classification

# create model
device = torch.device("cpu")
# net1 setup
input_dim1 = input_dim
output_dim1 = 2
num_hidden1 = treat_depth
hidden_dim1 = hidden_dim[:num_hidden1]
# net2 setup
input_dim2 = hidden_dim1[-1]
output_dim2 = output_dim
num_hidden2 = num_hidden - treat_depth
hidden_dim2 = hidden_dim[num_hidden1:num_hidden]
# define network
np.random.seed(seed)
torch.manual_seed(seed)
net1 = Net(num_hidden1, hidden_dim1, input_dim1, output_dim1)
net2 = Net(num_hidden2, hidden_dim2, input_dim2, output_dim2)
net1.to(device)
net2.to(device)

# load model parameter
net1.load_state_dict(torch.load(os.path.join(PATH, 'model1' + '.pt')))
net2.load_state_dict(torch.load(os.path.join(PATH, 'model2' + '.pt')))

# calculate loss (and accuaracy)
test_loss, correct = 0, 0
num_batches = len(test_data)
test_size = test_set.__len__()
with torch.no_grad():
    for batch, (ate, ey1, ey0, y, treat, x) in enumerate(test_data):
        pred_temp = net1.module_dict[str(0)](x)
        for layer_index in range(num_hidden1 - 1):
            pred_temp = net1.module_dict[str(layer_index + 1)](pred_temp)
        pred = net2.forward(pred_temp)

        test_loss += loss(pred, y).item()
        if regression_flag is False:
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

test_loss /= num_batches
print(f"Avg val loss: {test_loss:>8f} \n")
if regression_flag is False:
    correct /= test_size
    print(f"accuracy: {correct:>8f} \n")

# estimate EY1, EY0, and ATE (need more knowledge about causal inference)
