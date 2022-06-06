import torch
import torch.nn as nn
from network_replicate import Net
from data import SimulationData_Cont, SimulationData_Bin, acic_data
from torch.utils.data import DataLoader, random_split
from torchtext.data import to_map_style_dataset
from torch.optim import SGD
import numpy as np
import argparse
import os
import errno

parser = argparse.ArgumentParser(description='Running stonet')
parser.add_argument('--seed', default=1, type=int, help='set seed')

# regression flag
parser.add_argument('--regression', dest='regression_flag', action='store_true', help='true for regression')
parser.add_argument('--classification', dest='regression_flag', action='store_false', help='false for classification')
# data
parser.add_argument('--data_source', default='sim', type=str,
                    help='specify the name of the data, the other option is acic')

parser.add_argument('--whole_data', dest='subset', action='store_false',
                    help='for acic data with continuous outcome variable, use the whole data to train')
parser.add_argument('--subset', dest='subset', action='store_true',
                    help='for acic data with continuous outcome variable, use a subset of data to train')
parser.set_defaults(subset=False)

parser.add_argument('--data_name', default='speed', type=str,
                    help='data name of the acic data with binary outcome variable. The other option is epi')
# model
parser.add_argument('--layer', default=3, type=int, help='number of hidden layer (not including the treatment layer')
parser.add_argument('--unit', default=[64, 32, 16], type=int, nargs='+', help='number of hidden unit in each layer')
parser.add_argument('--sigma', default=[1e-6, 1e-5, 1e-4, 1e-3], type=float, nargs='+',
                    help='variance of each layer for the model')

# Training Setting
parser.add_argument('--nepoch', default=1000, type=int, help='total number of training epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch size for training')

parser.add_argument('--mh_step', default=1, type=int, help='number of SGHMC step for imputation')
parser.add_argument('--impute_lr', default=[0.0000001], type=float, nargs='+', help='step size in imputation')
parser.add_argument('--ita', default=0.5, type=float, help='friction coefficient for SGHMC')

parser.add_argument('--para_lr', default=[1e-4, 1e-5, 1e-6, 1e-7], type=float, nargs='+',
                    help='step size in parameter update')
parser.add_argument('--para_momentum', default=0.9, type=float, help='momentum parameter for parameter update')

args = parser.parse_args()

# imputation parameters
impute_lrs = args.impute_lr
ita = args.ita

# training parameters
batch_size = args.batch_size
para_lrs = args.para_lr
para_momentum = args.para_momentum
epochs = args.nepoch
regression_flag = args.regression_flag
seed = args.seed
device = torch.device("cpu")

# data
data_source = args.data_source
subset = args.subset
data_name = args.data_name

# settings for regression and classification task
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

# simulation dataset
if data_source == 'sim':
    if regression_flag is True:
        data = SimulationData_Cont(1, 1000)
        train_set, val_set = random_split(data, [500, 500],
                                          generator=torch.Generator().manual_seed(seed))
    else:
        data = SimulationData_Bin(3, 1000)
        train_set, val_set = random_split(data, [500, 500],
                                          generator=torch.Generator().manual_seed(seed))
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

train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=32)
val_data = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=32)

# get input dim and output dim
y_temp, treat, x_temp = next(iter(train_data))
input_dim = x_temp[0].size(dim=0)

# network parameters
num_hidden = args.layer
hidden_dim = args.unit
sigma_list = args.sigma
mh_step = args.mh_step

# define network
np.random.seed(seed)
torch.manual_seed(seed)
net = Net(num_hidden, hidden_dim, input_dim, output_dim)
net.to(device)

# match the dimension of training parameters
if len(impute_lrs) == 1 and num_hidden > 1:
    temp_impute_lrs = impute_lrs[0]
    impute_lrs = []
    for i in range(num_hidden):
        impute_lrs.append(temp_impute_lrs)

if len(para_lrs) == 1 and num_hidden > 1:
    temp_para_lrs = para_lrs[0]
    para_lrs = []
    for i in range(num_hidden + 1):
        para_lrs.append(temp_para_lrs)

if len(sigma_list) == 1 and num_hidden > 1:
    temp_sigma_list = sigma_list[0]
    sigma_list = []
    for i in range(num_hidden + 1):
        sigma_list.append(temp_sigma_list)

# path to save the result
if regression_flag is True:
    base_path = os.path.join('.', 'stonet', 'result', data_source, 'regression')
else:
    base_path = os.path.join('.', 'stonet', 'result', data_source, 'classification', data_name)
spec = str(impute_lrs) + '_' + str(para_lrs) + '_' + str(hidden_dim) + '_' + str(epochs)
PATH = os.path.join(base_path, spec)

if not os.path.isdir(PATH):
    try:
        os.makedirs(PATH)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(PATH):
            pass
        else:
            raise

# define optimizer
optimizer_list = []
for i in range(num_hidden + 1):
    optimizer_list.append(SGD(net.module_dict[str(i)].parameters(), lr=para_lrs[i], momentum=para_momentum))

# training the network
train_num_batches = len(train_data)
train_size = train_set.__len__()
val_num_batches = len(val_data)
val_size = val_set.__len__()

for epoch in range(epochs):
    print("Epoch" + str(epoch))
    for batch, (y, treat, x) in enumerate(train_data):
        batch_size = y.size(dim=0)
        hidden_list = net.backward_imputation(mh_step, impute_lrs, ita, loss_sum, sigma_list, x, y)
        for p in net.parameters():
            p.grad = None

        for layer_index in range(num_hidden + 1):
            adj_factor = batch_size/train_size
            hidden_likelihood = -adj_factor * net.likelihood("p", hidden_list, layer_index, loss_sum, sigma_list, x, y)

            optimizer = optimizer_list[layer_index]
            hidden_likelihood.backward()
            optimizer.step()

    train_loss, train_correct = 0, 0
    with torch.no_grad():
        for batch, (y, treat, x) in enumerate(train_data):
            pred = net.forward(x)
            train_loss += loss(pred, y).item()
            if regression_flag is False:
                train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= train_num_batches
    train_loss_path[epoch] = train_loss
    print(f"Avg train loss: {train_loss:>8f} \n")
    if regression_flag is False:
        train_correct /= train_size
        train_accuracy_path[epoch] = train_correct
        print(f"train accuracy: {train_correct:>8f} \n")

    # validation loop
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for y, treat, x in val_data:
            pred = net.forward(x)
            loss_new = loss(pred, y).item()
            val_loss += loss_new
            if regression_flag is False:
                val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= val_num_batches
    val_loss_path[epoch] = val_loss
    print(f"Avg val loss: {val_loss:>8f} \n")
    if regression_flag is False:
        val_correct /= val_size
        val_accuracy_path[epoch] = val_correct
        print(f"accuracy: {val_correct:>8f} \n")

    torch.save(net.state_dict(), os.path.join(PATH, 'model' + str(epoch)+'.pt'))

np.savetxt(os.path.join(PATH, 'train_loss.txt'), train_loss_path, fmt="%s")
np.savetxt(os.path.join(PATH, 'val_loss.txt'), val_loss_path, fmt="%s")
if regression_flag is False:
    np.savetxt(os.path.join(PATH, 'train_accuracy.txt'), train_accuracy_path, fmt="%s")
    np.savetxt(os.path.join(PATH, 'val_accuracy.txt'), val_accuracy_path, fmt="%s")
