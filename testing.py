import torch
import torch.nn as nn
from network import Net
from data import acic_data_test
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
import numpy as np
import os
import argparse
import pickle
import errno


parser = argparse.ArgumentParser(description='Testing causal stonet')
parser.add_argument('--seed', default=1, type=int, help='set seed')
parser.add_argument('--regression', dest='regression_flag', action='store_true', help='true for regression')
parser.add_argument('--classification', dest='regression_flag', action='store_false', help='false for classification')

parser.add_argument('--layer', default=2, type=int, help='number of hidden layer (not including the treatment layer')
parser.add_argument('--unit', default=[20, 15], type=int, nargs='+', help='number of hidden unit in each layer')
parser.add_argument('--sigma', default=[0.001, 0.0001, 0.00001], type=float, nargs='+',
                    help='variance of each layer for the model')
parser.add_argument('--depth', default=1, type=int, help='number of layers before the treatment layer')
parser.add_argument('--treat_node', default=1, type=int, help='the position of the treatment variable')

parser.add_argument('--impute_lr', default=[0.0000001], type=float, nargs='+', help='step size in imputation')
parser.add_argument('--para_lr', default=[0.0001, 0.0000001], type=float, nargs='+',
                    help='step size in parameter update')

parser.add_argument('--batch_size', default=50, type=int, help='batch size for training')
parser.add_argument('--path', type=str, help='folder that saves the model parameters')

args = parser.parse_args()

seed = args.seed
regression_flag = args.regression_flag
num_hidden = args.layer
hidden_dim = args.unit
sigma_list = args.sigma
treat_depth = args.depth
treat_node = args.treat_node
batch_size = args.batch_size
PATH = args.path
impute_lrs = args.impute_lr
para_lrs = args.para_lr

# load data
if regression_flag is True:
    test_set = to_map_style_dataset(acic_data_test('cont'))
else:
    test_set = to_map_style_dataset(acic_data_test('bin'))
test_data = DataLoader(test_set, batch_size=batch_size)

# get input dim and output dim
_, _, _, _, _, x_temp = next(iter(test_data))
input_dim = x_temp[0].size(dim=0)

# settings for loss functions
if regression_flag:
    loss = nn.MSELoss()
    output_dim = 1
    test_loss_path = []
else:
    loss = nn.CrossEntropyLoss()
    output_dim = 2  # for binary classification; change accordingly for multi-level classification
    test_loss_path = []
    test_accuracy_path = []

# define network
np.random.seed(seed)
torch.manual_seed(seed)
net = Net(num_hidden, hidden_dim, input_dim, output_dim, treat_depth, treat_node)
device = torch.device("cpu")
net.to(device)

# define path to read data
if regression_flag is True:
    base_path = os.path.join('.', 'result', 'acic', 'regression')
else:
    base_path = os.path.join('.', 'result', 'acic', 'classification')
spec = str(impute_lrs[0]) + '_' + str(para_lrs[0]) + '_' + str(hidden_dim)
PATH = os.path.join(base_path, spec)

if not os.path.isdir(PATH):
    try:
        os.makedirs(PATH)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(PATH):
            pass
        else:
            raise

# load model parameter
net.load_state_dict(torch.load(os.path.join(PATH, 'model' + '.pt')))

# calculate loss (and accuracy)
test_loss, correct = 0, 0
with torch.no_grad():
    for batch, (ate, ey1, ey0, y, treat, x) in enumerate(test_data):
        num_sample = y.size(dim=0)

        pred = net.forward(x)
        test_loss = loss(pred, y).item()
        test_loss_path[batch] = test_loss
        print(f"Avg test loss: {test_loss:>8f} \n")

        if regression_flag is False:
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= num_sample
            test_accuracy_path[batch] = correct
            print(f"test accuracy: {correct:>8f} \n")

# estimate EY1, EY0, and ATE (need more knowledge about causal inference)

if regression_flag:
    filename = PATH + 'training_result.txt'
    f = open(filename, 'wb')
    pickle.dump(test_loss_path, f)
    f.close()
else:
    filename = PATH + 'training_result.txt'
    f = open(filename, 'wb')
    pickle.dump([test_loss_path,test_accuracy_path], f)
    f.close()
