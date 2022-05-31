import torch
import torch.nn as nn
from network import Net
from data import SimulationData_Cont, SimulationData_Bin, acic_data
from torch.utils.data import DataLoader, random_split
from torchtext.data import to_map_style_dataset
from torch.optim import SGD
import numpy as np
import os
import argparse
import errno


parser = argparse.ArgumentParser(description='Running stonet with treatment layer')
parser.add_argument('--seed', default=1, type=int, help='set seed')

parser.add_argument('--regression', dest='regression_flag', action='store_true', help='true for regression')
parser.add_argument('--classification', dest='regression_flag', action='store_false', help='false for classification')

# data
parser.add_argument('--data_source', default='sim', type=str,
                    help='specify the name of the data, the other option is acic')

parser.add_argument('--whole_data', dest='subset', action='store_false',
                    help='for acic data with continuous out come variable, use the whole data to train')
parser.add_argument('--subset', dest='subset', action='store_true',
                    help='for acic data with continuous out come variable, use a subset of data to train')
parser.set_defaults(subset=False)

parser.add_argument('--data_name', default='speed', type=str,
                    help='data name of the acic data with binary outcome variable. The other option is epi')

# model
parser.add_argument('--layer', default=2, type=int, help='number of hidden layer (not including the treatment layer')
parser.add_argument('--unit', default=[20, 15], type=int, nargs='+', help='number of hidden unit in each layer')
parser.add_argument('--sigma', default=[0.001, 0.0001, 0.00001], type=float, nargs='+',
                    help='variance of each layer for the model')
parser.add_argument('--depth', default=1, type=int, help='number of layers before the treatment layer')

# Training Setting
parser.add_argument('--nepoch', default=100, type=int, help='total number of training epochs')
parser.add_argument('--batch_size', default=50, type=int, help='batch size for training')

parser.add_argument('--mh_step', default=1, type=int, help='number of SGMCMC step for imputation')
parser.add_argument('--impute_lr', default=[0.0000001], type=float, nargs='+', help='step size in imputation')
parser.add_argument('--ita', default=0.5, type=float, help='friction coefficient for SGHMC')

parser.add_argument('--para_lr', default=[0.0001, 0.0000001], type=float, nargs='+',
                    help='step size in parameter update')
parser.add_argument('--para_momentum', default=0.9, type=float, help='momentum parameter for parameter update')

args = parser.parse_args()

# training parameters
impute_lrs = args.impute_lr
ita = args.ita
batch_size = args.batch_size
para_lrs = args.para_lr
para_momentum = args.para_momentum
epochs = args.nepoch
regression_flag = args.regression_flag
data_source = args.data_source
subset = args.subset
data_name = args.data_name
seed = args.seed
device = torch.device("cpu")
mh_step = args.mh_step

# settings for loss functions
treat_loss_sum = nn.CrossEntropyLoss(reduction='sum')
if regression_flag:
    loss = nn.MSELoss()
    loss_sum = nn.MSELoss(reduction='sum')
    output_dim = 1
    train_loss_path = np.zeros(epochs)
    val_loss_path = np.zeros(epochs)
else:
    loss = nn.CrossEntropyLoss()
    loss_sum = nn.CrossEntropyLoss(reduction='sum')
    output_dim = 2  # for binary classification; change accordingly for multi-level classification
    train_loss_path = np.zeros(epochs)
    val_loss_path = np.zeros(epochs)
    train_accuracy_path = np.zeros(epochs)
    val_accuracy_path = np.zeros(epochs)

# load data for training
if data_source == 'sim':
    if regression_flag is True:
        train_set = SimulationData_Cont(1, 500)
        val_set = SimulationData_Cont(2, 500)
    else:
        train_set = SimulationData_Bin(3, 500)
        val_set = SimulationData_Bin(4, 500)
if data_source == 'acic':
    if regression_flag is True:
        data = to_map_style_dataset(acic_data('cont', subset, data_name))
        data_size = data.__len__()
        train_size = int(data_size * 0.8)
        train_set, val_set = random_split(data, [train_size, data_size - train_size],
                                          generator=torch.Generator().manual_seed(seed))
    else:
        data = to_map_style_dataset(acic_data('bin', subset, data_name))
        data_size = data.__len__()
        train_size = int(data_size * 0.8)
        train_set, val_set = random_split(data, [train_size, data_size - train_size],
                                          generator=torch.Generator().manual_seed(seed))

train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_data = DataLoader(val_set, batch_size=batch_size, shuffle=True)

# get input dim and output dim
_, _, x_temp = next(iter(train_data))
input_dim = x_temp[0].size(dim=0)

# network setup
num_hidden = args.layer
treat_depth = args.depth
hidden_dim = args.unit
sigma_list = args.sigma
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

# match the dimension of training parameters
if len(impute_lrs) == 1 and num_hidden > 1:
    temp_impute_lrs = impute_lrs[0]
    impute_lrs = []
    for i in range(num_hidden):
        impute_lrs.append(temp_impute_lrs)

if len(para_lrs) == 1 and num_hidden > 1:
    temp_para_lrs = para_lrs[0]
    para_lrs = []
    for i in range(num_hidden):
        para_lrs.append(temp_para_lrs)

if len(sigma_list) == 1 and num_hidden > 1:
    temp_sigma_list = sigma_list[0]
    sigma_list = []
    for i in range(num_hidden):
        sigma_list.append(temp_sigma_list)

# match the training parameters to the networks
# net1
sigma_list1 = sigma_list[:num_hidden1 + 1]
impute_lrs1 = impute_lrs[:num_hidden1]
para_lrs1 = para_lrs[:num_hidden1 + 1]
# net2
sigma_list2 = sigma_list[num_hidden1:num_hidden + 1]
impute_lrs2 = impute_lrs[num_hidden1:num_hidden]
para_lrs2 = para_lrs[num_hidden1:num_hidden + 1]

# define optimizer for parameter update
optimizer_list = []
for i in range(num_hidden1):
    optimizer_list.append(SGD(net1.module_dict[str(i)].parameters(), lr=para_lrs1[i], momentum=para_momentum))
for i in range(num_hidden2):
    optimizer_list.append(SGD(net2.module_dict[str(i)].parameters(), lr=para_lrs2[i], momentum=para_momentum))

# path to save the result
if regression_flag is True:
    base_path = os.path.join('.', 'result', data_source, 'regression')
else:
    base_path = os.path.join('.', 'result', data_source, 'classification')
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


for epoch in range(epochs):
    print("Epoch" + str(epoch))
    # train loop
    for batch, (y, treat, x) in enumerate(train_data):
        # backward imputation
        for step in range(mh_step):
            hidden_list1 = net1.backward_imputation(1, impute_lrs1, ita, treat_loss_sum, sigma_list1, x, treat)
            hidden_list2 = net2.backward_imputation(1, impute_lrs2, ita, loss_sum, sigma_list2, hidden_list1[-1], y)

        # parameter update
        for p1 in net1.parameters():
            if p1.grad is not None:
                p1.grad.zero_()

        for p1 in net2.parameters():
            if p1.grad is not None:
                p1.grad.zero_()

        adj_factor = batch_size/input_dim
        for layer_index in range(num_hidden1):
            hidden_likelihood1 = -adj_factor * net1.likelihood("p", hidden_list1, layer_index, treat_loss_sum,
                                                               sigma_list1, x, treat)
            optimizer = optimizer_list[layer_index]
            optimizer.zero_grad()
            hidden_likelihood1.backward()
            optimizer.step()

        for layer_index in range(num_hidden2):
            hidden_likelihood2 = -adj_factor * net2.likelihood("p", hidden_list2, layer_index, loss_sum, sigma_list2,
                                                               hidden_list1[-1], y)
            optimizer = optimizer_list[num_hidden1 + layer_index]
            optimizer.zero_grad()
            hidden_likelihood2.backward()
            optimizer.step()

    pred_temp = net1.module_dict[str(0)](x)
    for layer_index in range(num_hidden1 - 1):
        pred_temp = net1.module_dict[str(layer_index + 1)](pred_temp)
    pred = net2.forward(pred_temp)

    train_loss = loss(pred, y).item()
    train_loss_path[epoch] = train_loss

    print(f"Avg train loss: {train_loss:>8f} \n")

    # validation loop
    val_loss, correct = 0, 0
    num_batches = len(val_data)
    val_size = val_set.__len__()
    with torch.no_grad():
        for batch, (y, treat, x) in enumerate(val_data):
            pred_temp = net1.module_dict[str(0)](x)
            for layer_index in range(num_hidden1 - 1):
                pred_temp = net1.module_dict[str(layer_index + 1)](pred_temp)
            pred = net2.forward(pred_temp)

            val_loss += loss(pred, y).item()
            if regression_flag is False:
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    val_loss_path[epoch] = val_loss
    print(f"Avg val loss: {val_loss:>8f} \n")
    if regression_flag is False:
        correct /= val_size
        val_accuracy_path[epoch] = correct
        print(f"accuracy: {correct:>8f} \n")

torch.save(net1.state_dict(), os.path.join(PATH, 'model1' + '.pt'))
torch.save(net2.state_dict(), os.path.join(PATH, 'model2' + '.pt'))
