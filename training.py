import torch
import torch.nn as nn
from network import Net
from data import SimulationData_Cont, SimulationData_Bin
from torch.utils.data import DataLoader
from torch.optim import SGD
import numpy as np

# need to specify path to save final parameters

# training parameters
impute_lrs = [0.0000001]
ita = 0.5
batch_size = 50
para_lrs = [0.0001, 0.0000001]
para_momentum = 0.9
epochs = 10
regression_flag = True
seed = 1
device = torch.device("cpu")

# settings for regression and classification task
if regression_flag:
    loss = nn.MSELoss()
    loss_sum = nn.MSELoss(reduction='sum')
    output_dim = 1
    train_loss_path = np.zeros(epochs)
    test_loss_path = np.zeros(epochs)
else:
    loss = nn.CrossEntropyLoss()
    loss_sum = nn.CrossEntropyLoss(reduction='sum')
    output_dim = 2  # for binary classification; change accordingly for multi-level classification
    train_loss_path = np.zeros(epochs)
    test_loss_path = np.zeros(epochs)
    train_accuracy_path = np.zeros(epochs)
    test_accuracy_path = np.zeros(epochs)

# simulation dataset
if regression_flag is True:
    train_set = SimulationData_Cont(1, 500)
    test_set = SimulationData_Cont(2, 500)
else:
    train_set = SimulationData_Bin(3, 500)
    test_set = SimulationData_Bin(4, 500)

train_data = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)
# setting num_workers > 0 may hasve some potential problems
# check https://pytorch.org/docs/stable/data.html when loading ACIC data
test_data = DataLoader(test_set, batch_size=batch_size, num_workers=8, shuffle=True)

# get input dim and output dim
y_temp, x_temp = next(iter(train_data))
input_dim = x_temp[0].size(dim=0)

# network parameters
num_hidden = 1
hidden_dim = [20]
sigma_list = [0.001, 0.0001]
mh_step = 1

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
    for i in range(num_hidden):
        para_lrs.append(temp_para_lrs)

if len(sigma_list) == 1 and num_hidden > 1:
    temp_sigma_list = sigma_list[0]
    sigma_list = []
    for i in range(num_hidden):
        sigma_list.append(temp_sigma_list)

# define optimizer
optimizer_list = []
for i in range(num_hidden):
    optimizer_list.append(SGD(net.module_dict[str(i)].parameters(), lr=para_lrs[i], momentum=para_momentum))

for epoch in range(epochs):
    print("Epoch" + str(epoch))
    # train loop
    for batch, (y, x) in enumerate(train_data):
        hidden_list = net.backward_imputation(mh_step, impute_lrs, ita, loss_sum, sigma_list, x, y)
        for p in net.parameters():
            if p.grad is not None:
                p.grad.zero_()

        for layer_index in range(num_hidden):
            adj_factor = batch_size/input_dim
            hidden_likelihood = -adj_factor * net.likelihood("p", hidden_list, layer_index, loss_sum, sigma_list, x, y)
            optimizer = optimizer_list[layer_index]
            optimizer.zero_grad()
            hidden_likelihood.backward()
            optimizer.step()

    pred = net.forward(x)
    train_loss = loss(pred, y).item()
    train_loss_path[epoch] = train_loss

    print(f"Avg train loss: {train_loss:>8f} \n")

    # test loop
    test_loss, correct = 0, 0
    num_batches = len(test_data)
    size = len(test_data.dataset)
    with torch.no_grad():
        for batch, (y, x) in enumerate(test_data):
            pred = net.forward(x)
            test_loss += loss(pred, y).item()
            if regression_flag is False:
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    test_loss_path[epoch] = test_loss
    print(f"Avg test loss: {test_loss:>8f} \n")
    if regression_flag is False:
        correct /= size
        test_accuracy_path[epoch] = correct
        print(f"accuracy: {correct:>8f} \n")
